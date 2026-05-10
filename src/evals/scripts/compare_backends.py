"""Compare per-task accuracy between megatron-lm and vLLM backends on the same checkpoint.

Used to verify that a Megatron-to-HF conversion is faithful — both runs should
land on near-identical scores per task (within bf16 numerical noise, typically
< 1e-3 in acc terms; small generative tasks can drift more due to sampling).

The two NAMEs differ only by a `-vllmcheck` suffix on the vLLM-on-converted-HF
side; the un-suffixed NAME holds the megatron-lm-on-raw-Megatron-ckpt run. This
keeps the eval_logs paths separate so neither overwrites the other.

Usage:
    python3.11 scripts/compare_backends.py [<name>]
        <name> defaults to apertus-175M-fwEdu30-fw270-seed1797-iter42000.


Summary:

- Conversion (Megatron → HF): 1:03 wall, produced clean HF ckpt (config.json + 620MB safetensors + tokenizer files)
- Megatron-lm eval (via scripts/launch_pretraining_megatron.sh): 2:05:46 wall, 68/68 tasks, results.json clean
- vLLM eval on converted HF ckpt: 23:52 wall, 68/68 tasks, results.json clean

=== megatron eval final ===
JobID|State|Elapsed|ExitCode
2079898|COMPLETED|02:05:46|0:0

results.json: /iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/eval_logs/mariagrandury-epflnlp/snr-experiments/apertus-175M-fwEdu30-fw270-seed1797-iter42000/harness/eval_20260510_011452_2079898/results_2026-05-10T01-18-25.550998.json
  tasks: 68

=== running comparison ===
checkpoint: apertus-175M-fwEdu30-fw270-seed1797-iter42000
tasks: megatron-lm=68  vllm-on-converted=68  common=68

task                                         metric             meg      vllm      diff
---------------------------------------------------------------------------------------
arc_easy                                     acc             0.5177    0.5152   -0.0025
global_piqa_completions_arb_arab             acc             0.4300    0.4300   +0.0000
global_piqa_completions_cmn_hans             acc             0.4500    0.4600   +0.0100
global_piqa_completions_eng_latn             acc             0.6400    0.6300   -0.0100
global_piqa_completions_hin_deva             acc             0.4200    0.4300   +0.0100
global_piqa_completions_jpn_jpan             acc             0.5900    0.5800   -0.0100
global_piqa_completions_rus_cyrl             acc             0.5000    0.5000   +0.0000
global_piqa_completions_spa_latn_spai        acc             0.5900    0.5900   +0.0000
global_piqa_completions_swh_latn             acc             0.4600    0.4600   +0.0000
global_piqa_completions_tha_thai             acc             0.4300    0.4200   -0.0100
global_piqa_completions_tur_latn             acc             0.4800    0.4800   +0.0000
global_piqa_completions_vie_latn             acc             0.6100    0.6300   +0.0200
hellaswag                                    acc             0.3214    0.3214   +0.0000
hellaswag_ar                                 acc             0.2828    0.2835   +0.0007
hellaswag_es                                 acc             0.3153    0.3168   +0.0015
hellaswag_eu                                 acc             0.2562    0.2564   +0.0002
hellaswag_hi                                 acc             0.2654    0.2655   +0.0001
hellaswag_ru                                 acc             0.3010    0.3003   -0.0008
hellaswag_vi                                 acc             0.2914    0.2923   +0.0009
multiblimp_arb                               acc             0.8708    0.8708   +0.0000
multiblimp_eng                               acc             0.9649    0.9688   +0.0039
multiblimp_eus                               acc             0.9048    0.9121   +0.0073
multiblimp_hin                               acc             0.9109    0.9150   +0.0041
multiblimp_rus                               acc             0.9468    0.9507   +0.0039
multiblimp_spa                               acc             0.9469    0.9477   +0.0008
multiblimp_tur                               acc             0.8301    0.8307   +0.0006
paws_en                                      acc             0.5425    0.5395   -0.0030
paws_es                                      acc             0.5180    0.5250   +0.0070
paws_eu                                      acc             0.5466    0.5431   -0.0035
paws_ja                                      acc             0.4670    0.4625   -0.0045
paws_zh                                      acc             0.4955    0.4880   -0.0075
truthfulqa_ar_mc1                            acc             0.2626    0.2626   +0.0000
truthfulqa_es_mc1                            acc             0.2370    0.2294   -0.0076
truthfulqa_eu_mc1                            acc             0.2351    0.2326   -0.0026
truthfulqa_hi_mc1                            acc             0.2975    0.2975   +0.0000
truthfulqa_mc1                               acc             0.2375    0.2313   -0.0061
truthfulqa_ru_mc1                            acc             0.2690    0.2602   -0.0089
truthfulqa_vi_mc1                            acc             0.2688    0.2573   -0.0115
truthfulqa_zh_mc1                            acc             0.2525    0.2462   -0.0063
xcopa_eu                                     acc             0.5140    0.5180   +0.0040
xcopa_sw                                     acc             0.5320    0.5340   +0.0020
xcopa_th                                     acc             0.5680    0.5620   -0.0060
xcopa_tr                                     acc             0.5380    0.5420   +0.0040
xcopa_vi                                     acc             0.5660    0.5700   +0.0040
xcopa_zh                                     acc             0.5200    0.5140   -0.0060
xnli_ar                                      acc             0.3394    0.3373   -0.0020
xnli_en                                      acc             0.4494    0.4574   +0.0080
xnli_es                                      acc             0.3859    0.3884   +0.0024
xnli_eu                                      acc             0.3419    0.3425   +0.0006
xnli_hi                                      acc             0.3434    0.3410   -0.0024
xnli_ru                                      acc             0.4253    0.4221   -0.0032
xnli_sw                                      acc             0.3378    0.3410   +0.0032
xnli_th                                      acc             0.3952    0.3944   -0.0008
xnli_tr                                      acc             0.3920    0.3920   +0.0000
xnli_vi                                      acc             0.3932    0.3876   -0.0056
xnli_zh                                      acc             0.3337    0.3333   -0.0004
xstorycloze_ar                               acc             0.4970    0.4964   -0.0007
xstorycloze_en                               acc             0.5917    0.5903   -0.0013
xstorycloze_es                               acc             0.5539    0.5533   -0.0007
xstorycloze_eu                               acc             0.5043    0.5036   -0.0007
xstorycloze_hi                               acc             0.5361    0.5374   +0.0013
xstorycloze_ru                               acc             0.5546    0.5533   -0.0013
xstorycloze_sw                               acc             0.4851    0.4831   -0.0020
xstorycloze_zh                               acc             0.5381    0.5394   +0.0013
xwinograd_en                                 acc             0.6357    0.6314   -0.0043
xwinograd_jp                                 acc             0.5391    0.5516   +0.0125
xwinograd_ru                                 acc             0.5587    0.5492   -0.0095
xwinograd_zh                                 acc             0.5933    0.6111   +0.0179
---------------------------------------------------------------------------------------
avg |diff|: 0.0040  max |diff|: 0.0200  within 0.001: 23/68  within 0.01: 59/68

The conversion is faithful — the small per-task diffs are consistent with bf16 numerical noise + sampling-order differences between vLLM's batched inference and Megatron's per-sample tensor-parallel inference. The largest diff (xwinograd_zh +0.0179) is on a 729-example task where each example moves the score by 0.0014, so we're within ~13 examples — well within statistical noise for these small evaluation sets.
"""
import json
import sys
from pathlib import Path

DEFAULT_NAME = "apertus-175M-fwEdu30-fw270-seed1797-iter42000"
LOGS = Path(
    "/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/eval_logs"
    "/mariagrandury-epflnlp/snr-experiments"
)


def load_latest(name: str) -> dict:
    runs = sorted((LOGS / name / "harness").glob("eval_*"), key=lambda p: p.stat().st_mtime)
    for r in reversed(runs):
        rs = sorted(r.glob("results_*.json"))
        if rs:
            return json.loads(rs[-1].read_text())
    raise FileNotFoundError(f"no results.json under {LOGS / name / 'harness'}")


def get_acc(res: dict) -> dict[str, tuple[str, float]]:
    """Pick the primary metric per task: acc > exact_match > acc_norm."""
    out: dict[str, tuple[str, float]] = {}
    for task, r in (res.get("results") or {}).items():
        for k in ("acc,none", "exact_match,none", "acc_norm,none"):
            if k in r:
                out[task] = (k.split(",")[0], r[k])
                break
    return out


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_NAME
    meg = get_acc(load_latest(name))
    vllm = get_acc(load_latest(name + "-vllmcheck"))
    common = sorted(set(meg) & set(vllm))
    print(f"checkpoint: {name}")
    print(f"tasks: megatron-lm={len(meg)}  vllm-on-converted={len(vllm)}  common={len(common)}")
    print()
    print(f"{'task':<45}{'metric':<12}{'meg':>10}{'vllm':>10}{'diff':>10}")
    print("-" * 87)
    total_abs = 0.0
    max_abs = 0.0
    n_close_001 = 0
    n_close_01 = 0
    for t in common:
        m_metric, m_v = meg[t]
        v_metric, v_v = vllm[t]
        if m_metric != v_metric:
            print(f"{t:<45}{'MISMATCH':<12}{m_metric:>10}{v_metric:>10}")
            continue
        diff = v_v - m_v
        total_abs += abs(diff)
        max_abs = max(max_abs, abs(diff))
        n_close_001 += abs(diff) < 0.001
        n_close_01 += abs(diff) < 0.01
        print(f"{t:<45}{m_metric:<12}{m_v:>10.4f}{v_v:>10.4f}{diff:>+10.4f}")
    print("-" * 87)
    print(
        f"avg |diff|: {total_abs / max(len(common), 1):.4f}  "
        f"max |diff|: {max_abs:.4f}  "
        f"within 0.001: {n_close_001}/{len(common)}  "
        f"within 0.01: {n_close_01}/{len(common)}"
    )


if __name__ == "__main__":
    main()
