import os, re, gc, csv, time, argparse, subprocess, json
from datetime import datetime
from typing import List, Optional
import threading
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

SAFE_NAME_PATTERN = re.compile(r'[^-\w.]')
def _safe_name(name: str) -> str:
    return SAFE_NAME_PATTERN.sub('_', name)

# pharser 
def read_gpu_stat_once():
    try:
        cmd = ['nvidia-smi', '--query-gpu=index,temperature.gpu,power.draw', '--format=csv,noheader,nounits']
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        out = result.stdout.strip()
        
        readings = {}
        if out:
            for line in out.split('\n'):
                parts = line.split(',')
                if len(parts) >= 3:
                    try:
                        dev_id = int(parts[0].strip())
                        temp = int(parts[1].strip())
                        power = float(parts[2].strip())
                        readings[dev_id] = (temp, power)
                    except ValueError:
                        continue
        return readings
    except Exception:
        return {}


class GPUMonitor:
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
                readings = read_gpu_stat_once()
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
                plt.plot(g["elapsed_sec"], g["temperature_c"], label=f"gpu{int(dev)}")
        plt.xlabel("Elapsed (s)"); plt.ylabel("Temp (°C)"); plt.title("GPU Temperature"); plt.legend()
        p1 = os.path.join(out_dir, "gpu_temperature.png")
        plt.savefig(p1, dpi=150, bbox_inches="tight"); plt.close()

        # Power
        plt.figure()
        for dev, g in df.groupby("device_id"):
            if dev not in active_devs:
                continue
            if g["power_w"].notna().any():
                plt.plot(g["elapsed_sec"], g["power_w"], label=f"gpu{int(dev)}")
        plt.xlabel("Elapsed (s)"); plt.ylabel("Power (W)"); plt.title("GPU Power"); plt.legend()
        p2 = os.path.join(out_dir, "gpu_power.png")
        plt.savefig(p2, dpi=150, bbox_inches="tight"); plt.close()

        print(f"[Plot] Saved: {p1}\n[Plot] Saved: {p2}")

def bench(model_name: str,
          prompts: List[str],
          outdir: str,
          max_new_tokens: int = 512,
          max_model_len: Optional[int] = None,
          monitor: bool = True,
          monitor_devices: Optional[List[int]] = None,
          monitor_interval: float = 1.0,
          src_texts: Optional[List[str]] = None,
          gold_texts: Optional[List[str]] = None,
          plot_min_watt: float = 5.0):
    
    batch_size = 1                  

    # 모델 로드 (NVIDIA CUDA)
    print(f"[Info] Loading model {model_name} on CUDA...")
    print(f"[Info] Max Model Len: {max_model_len if max_model_len else 'Auto (Model Config)'}")

    llm = LLM(
        model=model_name,
        tensor_parallel_size=1,        
        max_num_seqs=batch_size,       
        max_model_len=max_model_len,   
        gpu_memory_utilization=0.90,   
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    sampling_params = SamplingParams(
        temperature=0.0,
        skip_special_tokens=True,
        stop_token_ids=[tokenizer.eos_token_id],
        max_tokens=max_new_tokens 
    )

    # 출력 경로 세팅
    base_name = os.path.basename(os.path.normpath(model_name))
    model_out = os.path.join(outdir, base_name)
    os.makedirs(model_out, exist_ok=True)
    safe_model = _safe_name(base_name).lstrip('.')

    # GPU 모니터
    mon = None
    if monitor:
        if monitor_devices is None:
            monitor_devices = [0] 
        mon = GPUMonitor(log_path=os.path.join(model_out, f"{safe_model}_gpu_monitor.csv"),
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
            
            conversation = [{"role":"user", "content":q}]
            chat = tokenizer.apply_chat_template(
                conversation, 
                add_generation_prompt=True,
                tokenize=False)
            
            input_tokens = tokenizer([chat], return_tensors="pt", padding=True)["input_ids"]
            input_len = int(input_tokens.shape[-1])

            # generate 전후의 Wall Clock Time 측정
            # 내부 metrics가 실패하더라도 전체 latency는 정확히 보장함
            torch.cuda.synchronize() # GPU 동기화 
            start_time = time.perf_counter()
            
            outputs = llm.generate(chat, sampling_params)
            
            torch.cuda.synchronize() # GPU 완료 대기
            end_time = time.perf_counter()
            
            # 전체 추론 시간 (Wall Clock)
            total_inference_time = end_time - start_time

            request_output = outputs[0]
            metrics = request_output.metrics
            cand = request_output.outputs[0]
            new_tok = len(cand.token_ids)
            pred_text = cand.text

            # 1. TTFT: vLLM 내부 metrics를 우선 사용. 없으면 0.0 처리 (Latency에 포함됨)
            if metrics and metrics.first_token_time is not None:
                # arrival_time은 요청이 큐에 들어간 시간. 
                # offline batch에서는 start_time과 거의 비슷하지만, 내부 로직을 따름
                ttft = metrics.first_token_time - metrics.arrival_time
            else:
                ttft = 0.0 
            
            # TTFT가 비정상적으로 크거나(Latency보다 큼) 음수면 보정
            if ttft > total_inference_time:
                ttft = total_inference_time
            if ttft < 0: 
                ttft = 0.0

            # 2. TPOT: (전체 시간 - TTFT) / (생성된 토큰 수 - 1)
            if new_tok > 1:
                tpot = (total_inference_time - ttft) / (new_tok - 1)
            else:
                tpot = 0.0

            per_prompt_results.append({
                "prompt_id": i + 1,
                "latency_sec": total_inference_time,
                "ttft_sec": ttft,
                "tpot_sec_per_token": tpot,
                "generated_tokens": new_tok,
                "input_tokens": input_len,
                "prompt_text": q[:100] + '...' if len(q) > 100 else q,
                "source_en": (src_texts[i] if (src_texts and i < len(src_texts)) else None),
                "gold_ko": (gold_texts[i] if (gold_texts and i < len(gold_texts)) else None),
                "pred_ko": pred_text,
            })

            all_total.append(total_inference_time)
            all_new_tok.append(new_tok)
            all_ttft.append(ttft)

            print(f"TTFT {ttft:.4f}s | Generated {new_tok} tokens in {total_inference_time:.4f}s (TPOT: {tpot:.4f} s/tok).")

        print("Finished measurement.")
    finally:
        if mon is not None:
            mon.stop()
            GPUMonitor.plot(os.path.join(model_out, f"{safe_model}_gpu_monitor.csv"), out_dir=model_out, min_watt=plot_min_watt)
        try:
            del llm
        except Exception:
            pass
        try:
            del tokenizer
        except Exception:
            pass
        gc.collect()
        torch.cuda.empty_cache() 
        time.sleep(0.3)
    
    if per_prompt_results:
        detailed_df = pd.DataFrame(per_prompt_results)
        detailed_csv_path = os.path.join(model_out, f"{safe_model}_per_prompt_metrics.csv") 
        detailed_df.to_csv(detailed_csv_path, index=False, encoding="utf-8-sig")

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
    
    avg_tpot = 0.0
    valid_tpot_count = 0
    for r in per_prompt_results:
        if r['generated_tokens'] > 1:
            avg_tpot += r['tpot_sec_per_token']
            valid_tpot_count += 1
    if valid_tpot_count > 0:
        avg_tpot /= valid_tpot_count

    tput = (tot_gen / tot_time) if tot_time > 0 else 0.0 

    results = {
        "model_name": model_name,
        "num_prompts": num,
        "total_generated_tokens": tot_gen,
        "total_inference_time": tot_time,
        "avg_latency_per_prompt": avg_lat,
        "avg_ttft": avg_ttft,
        "avg_tpot": avg_tpot,
        "overall_throughput_tokens_per_sec": tput,
        "hardware": "NVIDIA_RTX_5090"
    }

    df = pd.DataFrame([results])
    csv_path = os.path.join(model_out, f"{safe_model}_benchmark_summary_{int(time.time())}.csv") 
    df.to_csv(csv_path, index=False, encoding="utf-8")

    print("\n" + "="*20 + " Average Performance Results " + "="*20)
    print(f"[Summary]")
    print(f" - Model Tested :                      {results['model_name']}")
    print(f" - Hardware :                          {results['hardware']}")
    print(f" - Total Prompts Processed :           {results['num_prompts']}")
    print(f" - Total Tokens Generated :            {results['total_generated_tokens']} tokens")
    print("\n[Latency Metrics]")
    print(f" - Average Response Time per Prompt :  {results['avg_latency_per_prompt']:.4f} sec")
    print(f" - Average TTFT :                      {results['avg_ttft']:.4f} sec")
    print(f" - Average TPOT :                      {results['avg_tpot']:.4f} sec/token")
    print("\n[Throughput Metrics]")
    print(f" - Overall Throughput :                {results['overall_throughput_tokens_per_sec']:.2f} tokens/sec")
    print("="*63)
    print(f"[Saved Summary] {csv_path}")

def parse_args():
    p = argparse.ArgumentParser(description="Single-model NVIDIA GPU vLLM benchmark")
    p.add_argument("--model", required=True, help="모델 이름(허브 ID 또는 로컬 경로)")
    p.add_argument("--jsonl-file", required=True, help="JSONL 파일 경로")
    p.add_argument("--outdir", default="./results", help="결과 저장 루트")
    p.add_argument("--max-new-tokens", type=int, default=512)
    # [수정] max-model-len 인자 추가
    p.add_argument("--max-model-len", type=int, default=None, help="모델 Context 길이 (기본값: 모델 config 자동 따름)")
    p.add_argument("--monitor", action="store_true", help="nvidia-smi 기반 GPU 모니터 활성화")
    p.add_argument("--monitor-devices", default="0", help="예: 0 또는 0,1")
    p.add_argument("--monitor-interval", type=float, default=1.0)
    p.add_argument("--plot-min-watt", type=float, default=5.0)
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
        max_model_len=args.max_model_len, 
        monitor=args.monitor,
        monitor_devices=devices,
        monitor_interval=args.monitor_interval,
        src_texts=src_texts,
        gold_texts=gold_texts,
        plot_min_watt=args.plot_min_watt,
    )
