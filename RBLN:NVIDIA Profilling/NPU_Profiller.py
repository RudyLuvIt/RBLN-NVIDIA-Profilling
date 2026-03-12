#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, gc, csv, time, argparse, subprocess, json
from datetime import datetime
from typing import List, Optional
import threading
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")  # 헤드리스 환경
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from optimum.rbln import RBLNAutoModelForCausalLM

SAFE_NAME_PATTERN = re.compile(r'[^-\w.]')
def _safe_name(name: str) -> str:
    return SAFE_NAME_PATTERN.sub('_', name)


# ---------- StoppingCriteria: 첫 토큰 도달 시각 기록 ttft----------
class FirstTokenTimer(StoppingCriteria):
    def __init__(self):
        super().__init__()
        self.start_time: Optional[float] = None
        self.ttft: Optional[float] = None
    def __call__(self, input_ids, scores, **kwargs):
        if self.ttft is None and self.start_time is not None:
            self.ttft = time.monotonic() - self.start_time
        return False


# ---------- 유틸: rbln-stat 파싱 ----------
STAT_PATTERN = re.compile(r"\|\s*(\d+)\s*\|.+?\|\s*(\d+)C\s*\|\s*([\d\.]+)W\s*\|")
def read_npu_stat_once():
    try:
        result = subprocess.run(['rbln-stat'], capture_output=True, text=True, check=False)
        out = result.stdout
        readings = {}
        for m in STAT_PATTERN.finditer(out):
            dev_id = int(m.group(1))
            temp = int(m.group(2))
            power = float(m.group(3))
            readings[dev_id] = (temp, power)
        return readings
    except Exception:
        return {}


# ---------- NPU 모니터 ----------
class NPUMonitor:
    def __init__(self, log_path: str, device_ids, interval_sec: float = 1.0):
        self.log_path = log_path
        self.device_ids = list(device_ids)
        self.interval = interval_sec
        self._thread = None
        self._stop = threading.Event()
        self._start_monotonic = None
        self._csv = None

    def start(self):
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        file_exists = os.path.exists(self.log_path)
        self._csv = open(self.log_path, "a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._csv)
        if not file_exists:
            self._writer.writerow(["timestamp_iso", "elapsed_sec", "device_id", "temperature_c", "power_w"])
        self._start_monotonic = time.monotonic()

        def _loop():
            while not self._stop.is_set():
                now_iso = datetime.now().isoformat(timespec="seconds")
                elapsed = time.monotonic() - self._start_monotonic
                readings = read_npu_stat_once()
                for dev in self.device_ids:
                    if dev in readings:
                        temp, power = readings[dev]
                        row = [now_iso, f"{elapsed:.2f}", dev, temp, power]
                    else:
                        row = [now_iso, f"{elapsed:.2f}", dev, "", ""]
                    try:
                        self._writer.writerow(row)
                    except Exception:
                        pass
                try:
                    self._csv.flush()
                except Exception:
                    pass
                time.sleep(self.interval)

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()
        print(f"[Monitor] Started. -> {self.log_path} (devices={self.device_ids})")

    def stop(self):
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=5)
        try:
            self._csv.close()
        except Exception:
            pass
        self._thread = None
        print("[Monitor] Stopped.")

    @staticmethod
    def plot(log_path: str, out_dir: str, min_watt: float = 5.0):
        if not os.path.exists(log_path):
            print(f"[Plot] Log not found: {log_path}")
            return
        df = pd.read_csv(log_path)
        for col in ["elapsed_sec", "temperature_c", "power_w", "device_id"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "elapsed_sec" in df.columns and df["elapsed_sec"].isna().all() and "timestamp_iso" in df.columns:
            df["timestamp_iso"] = pd.to_datetime(df["timestamp_iso"], errors="coerce")
            t0 = df["timestamp_iso"].min()
            df["elapsed_sec"] = (df["timestamp_iso"] - t0).dt.total_seconds()

        os.makedirs(out_dir, exist_ok=True)

        # 활성 장치만 선택: 95퍼센타일 전력이 min_watt 이상이거나 변동폭이 2W 이상
        active_devs = []
        for dev, g in df.groupby("device_id"):
            pw = g["power_w"].fillna(0)
            try:
                p95 = float(pw.quantile(0.95))
            except Exception:
                p95 = 0.0
            if (p95 >= float(min_watt)) or ((pw.max() - pw.min()) > 2.0):
                active_devs.append(dev)
        if not active_devs:
            active_devs = sorted([d for d in df["device_id"].dropna().unique()])

        # Temperature
        plt.figure()
        for dev, g in df.groupby("device_id"):
            if dev not in active_devs:
                continue
            if g["temperature_c"].notna().any():
                plt.plot(g["elapsed_sec"], g["temperature_c"], label=f"dev{int(dev)}")
        plt.xlabel("Elapsed (s)"); plt.ylabel("Temp (°C)"); plt.title("NPU Temperature"); plt.legend()
        p1 = os.path.join(out_dir, "npu_temperature.png")
        plt.savefig(p1, dpi=150, bbox_inches="tight"); plt.close()

        # Power
        plt.figure()
        for dev, g in df.groupby("device_id"):
            if dev not in active_devs:
                continue
            if g["power_w"].notna().any():
                plt.plot(g["elapsed_sec"], g["power_w"], label=f"dev{int(dev)}")
        plt.xlabel("Elapsed (s)"); plt.ylabel("Power (W)"); plt.title("NPU Power"); plt.legend()
        p2 = os.path.join(out_dir, "npu_power.png")
        plt.savefig(p2, dpi=150, bbox_inches="tight"); plt.close()

        print(f"[Plot] Saved: {p1}\n[Plot] Saved: {p2}")


# ---------- 벤치 로직(단일 모델) ----------
def need_token_type_ids(model) -> bool:
    cfg = getattr(model, "config", None)
    return bool(cfg and getattr(cfg, "type_vocab_size", 0) > 1)

def build_inputs(tokenizer, query: str, provide_token_type_ids: bool):
    conversation = [{"role": "user", "content": query}]
    chat = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    model_inputs = tokenizer([chat], return_tensors="pt", padding=True)
    if provide_token_type_ids:
        if "token_type_ids" not in model_inputs or model_inputs["token_type_ids"] is None:
            model_inputs["token_type_ids"] = model_inputs["input_ids"].new_zeros(model_inputs["input_ids"].shape)
    else:
        model_inputs.pop("token_type_ids", None)
    return model_inputs

def bench(model_name: str,
          prompts: List[str],
          outdir: str,
          max_new_tokens: int = 512,
          monitor: bool = True,
          monitor_devices: Optional[List[int]] = None,
          monitor_interval: float = 1.0,
          src_texts: Optional[List[str]] = None,
          gold_texts: Optional[List[str]] = None,
          plot_min_watt: float = 5.0):

    # 모델 로드
    llm = RBLNAutoModelForCausalLM.from_pretrained(model_id=model_name, export=False)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 출력 경로
    model_out = os.path.join(outdir, model_name)
    os.makedirs(model_out, exist_ok=True)
    safe_model = _safe_name(model_name)

    # NPU 모니터
    mon = None
    if monitor:
        if monitor_devices is None:
            monitor_devices = list(range(8))
        mon = NPUMonitor(log_path=os.path.join(model_out, f"{safe_model}_npu_monitor.csv"),
                         device_ids=monitor_devices,
                         interval_sec=monitor_interval)
        mon.start()

    # 측정
    all_total, all_new_tok, all_ttft = [], [], []
    torch.backends.cuda.matmul.allow_tf32 = True if torch.cuda.is_available() else False

    per_prompt_results = []

    try:
        for i, q in enumerate(prompts):
            print(f"\n--- Processing prompt {i+1}/{len(prompts)} ---")
            provide_tti = need_token_type_ids(llm)
            inputs = build_inputs(tokenizer, q, provide_tti)
            input_len = int(inputs["input_ids"].shape[-1])

            gen_kwargs = dict(
                input_ids=inputs["input_ids"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=False,
            )
            if "attention_mask" in inputs:
                gen_kwargs["attention_mask"] = inputs["attention_mask"]
            if "token_type_ids" in inputs:
                gen_kwargs["token_type_ids"] = inputs["token_type_ids"]

            timer = FirstTokenTimer() 
            timer.start_time = time.monotonic()
            t0 = timer.start_time  ## 여기가 시작 
            with torch.inference_mode():
                outputs = llm.generate(**gen_kwargs, stopping_criteria=StoppingCriteriaList([timer]))
            t1 = time.monotonic()  ## 여기가 종료

            total = t1 - t0
            if timer.ttft is not None:
                ttft = timer.ttft
                print("timer.ttft=")
            else:
                ttft = total
                print("total=")
            decode_time = max(1e-9, total - ttft) ## 디코딩 구간(tpot 구간)
            seq = outputs.sequences
            new_tok = int(seq.shape[-1] - input_len)
            try:
                pred_text = tokenizer.decode(seq[0][input_len:], skip_special_tokens=True)
            except Exception:
                pred_text = ""

            tpot = decode_time / new_tok if new_tok > 0 else 0.0

            per_prompt_results.append({
                "prompt_id": i + 1,
                "latency_sec": total,
                "ttft_sec": ttft,
                "tpot_sec_per_token": tpot,
                "generated_tokens": new_tok,
                "input_tokens": input_len,
                "prompt_text": q[:100] + '...' if len(q) > 100 else q,
                "source_en": (src_texts[i] if (src_texts and i < len(src_texts)) else None),
                "gold_ko": (gold_texts[i] if (gold_texts and i < len(gold_texts)) else None),
                "pred_ko": pred_text,
            })

            # 요약 계산을 위한 누적
            all_total.append(total)
            all_new_tok.append(new_tok)
            all_ttft.append(ttft)

            print(f"TTFT {ttft:.4f}s | Generated {new_tok} tokens in {total:.4f}s (TPOT: {tpot:.4f} s/tok).")

        print("Finished measurement.")
    finally: # 모니터 중단, csv를 읽어서 그래프 2장 저장
        if mon is not None:
            mon.stop()
            NPUMonitor.plot(os.path.join(model_out, f"{safe_model}_npu_monitor.csv"), out_dir=model_out, min_watt=plot_min_watt)

        try:
            del llm
        except Exception:
            pass
        try:
            del tokenizer
        except Exception:
            pass
        gc.collect()
        time.sleep(0.3)

    if per_prompt_results:
        detailed_df = pd.DataFrame(per_prompt_results)
        detailed_csv_path = os.path.join(model_out, f"{safe_model}_per_prompt_metrics.csv") 
        #latency_sec, ttft_sec, tpot_sec_per_token, generated_tokens, input_tokens, pred_ko
        detailed_df.to_csv(detailed_csv_path, index=False, encoding="utf-8-sig")

        # predictions triplet
        pred_jsonl = os.path.join(model_out, f"{safe_model}_predictions.jsonl")
        with open(pred_jsonl, "w", encoding="utf-8") as w:
            for row in per_prompt_results:
                rec = {k: row.get(k) for k in ["prompt_id","source_en","gold_ko","pred_ko"]}
                w.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"\n[Saved Detailed Metrics] {detailed_csv_path}")
        print(f"[Saved Predictions ] {pred_jsonl}")

    num = len(all_total)
    tot_gen = sum(all_new_tok)
    tot_time = sum(all_total)
    avg_lat = (tot_time / num) if num > 0 else 0.0
    avg_ttft = (sum(all_ttft) / num) if num > 0 else 0.0
    tot_decode = max(1e-9, tot_time - sum(all_ttft))
    tput = (tot_gen / tot_decode) if tot_decode > 0 else 0.0
    avg_tpot = (tot_decode / tot_gen) if tot_gen > 0 else 0.0

    results = {
        "model_name": model_name,
        "num_prompts": num,
        "total_generated_tokens": tot_gen,
        "total_inference_time": tot_time,
        "avg_latency_per_prompt": avg_lat,
        "avg_ttft": avg_ttft,
        "avg_tpot": avg_tpot,
        "overall_throughput_tokens_per_sec": tput,
    }

    df = pd.DataFrame([results])
    csv_path = os.path.join(model_out, f"{safe_model}_benchmark_summary_{int(time.time())}.csv") 
    #avg_latency_per_prompt, avg_ttft, avg_tpot, overall_throughput_tokens_per_sec
    df.to_csv(csv_path, index=False, encoding="utf-8")

    print("\n" + "="*20 + " Average Performance Results " + "="*20)
    print(f"[Summary]")
    print(f" - Model Tested :                      {results['model_name']}")
    print(f" - Total Prompts Processed :           {results['num_prompts']}")
    print(f" - Total Tokens Generated :            {results['total_generated_tokens']} tokens")
    print("\n[Latency Metrics]")
    print(f" - Average Response Time per Prompt :  {results['avg_latency_per_prompt']:.4f} sec")
    print(f" - (Note) Average Time To First Token (TTFT) : {results['avg_ttft']:.4f} sec")
    print(f" - Average Time Per Output Token (Avg TPOT) :  {results['avg_tpot']:.4f} sec/token")
    print("\n[Throughput Metrics]")
    print(f" - Overall Throughput (Tokens/Sec) :       {results['overall_throughput_tokens_per_sec']:.2f} tokens/sec")
    print("="*63)
    print(f"[Saved Summary] {csv_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Single-model RBLN LLM benchmark (JSONL prompts)")
    p.add_argument("--model", required=True, help="모델 이름(허브 ID 또는 로컬 경로)")
    p.add_argument("--jsonl-file", required=True, help="JSONL 파일 경로 (prompt 또는 source/target 사용)")
    p.add_argument("--outdir", default="./results", help="결과 저장 루트")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--monitor", action="store_true", help="rbln-stat 기반 NPU 모니터 활성화")
    p.add_argument("--monitor-devices", default="0-7", help="예: 0-3 또는 0,2,4")
    p.add_argument("--monitor-interval", type=float, default=1.0)
    p.add_argument("--plot-min-watt", type=float, default=5.0, help="95퍼센타일 전력이 이 값 미만인 장치는 그래프에서 숨김")
    return p.parse_args()


def parse_devices(s: str):
    s = s.strip()
    if "-" in s:
        a, b = s.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",") if x.strip() != ""]


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # JSONL 로드
    prompts: List[str] = []
    src_texts: List[Optional[str]] = []
    gold_texts: List[Optional[str]] = []
    with open(args.jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("prompt"):
                prompts.append(obj["prompt"])
                src_texts.append(None)
                gold_texts.append(None)
                continue
            if obj.get("source"):
                src = obj.get("source", "")
                gold = obj.get("target") or obj.get("original_target")
                pr = f"다음 영어 문장을 한국어로 번역하세요.\n\n{src}"
                prompts.append(pr)
                src_texts.append(src)
                gold_texts.append(gold)
                continue
            ins = obj.get("instruction", "")
            txt = obj.get("text", "")
            merged = (ins + ("\n\n" if ins and txt else "") + txt).strip()
            if merged:
                prompts.append(merged)
                src_texts.append(None)
                gold_texts.append(None)

    if not prompts:
        raise ValueError(f"JSONL에서 사용할 프롬프트를 찾지 못했습니다: {args.jsonl_file}")

    devices = parse_devices(args.monitor_devices)
    bench(
        model_name=args.model,
        prompts=prompts,
        outdir=args.outdir,
        max_new_tokens=args.max_new_tokens,
        monitor=args.monitor,
        monitor_devices=devices,
        monitor_interval=args.monitor_interval,
        src_texts=src_texts,
        gold_texts=gold_texts,
        plot_min_watt=args.plot_min_watt,
    )
