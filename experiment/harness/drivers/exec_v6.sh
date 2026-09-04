#!/bin/bash
cd ~/integer_alibi
exec > logs/exec_v6.log 2>&1
set -x
PIN=vllm/vllm-openai@sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967
# 1) clean environment: digest-pinned build plus the three test suites
docker build -q -f Dockerfile.test -t integer-alibi-test:v6 . > .build_id
{
  echo "git_sha: 3dbd6bf"
  echo "base_image: python:3.12-slim@sha256:229a2c5bfa27522db7815ea81f9bed70af17ccb9de9fc7ad142b1877b5830d36"
  echo "built_image_id: $(cat .build_id)"
  echo "platform: $(uname -a)"
  echo "command: docker run --rm integer-alibi-test:v6"
  docker run --rm integer-alibi-test:v6; echo "EXIT_clean_env=$?"
} > clean_env_test_record.txt 2>&1
# 2) refit v6 (neutral key plus metadata)
./.venv/bin/python3 w3_p2_refit_v4.py w3_p2_raw_20seeds.json w3_p2_fits_v6_20seeds.json
# 3) checkpoint manifest v2(P2)
./.venv/bin/python3 - <<'PYEOF'
import glob, hashlib, json
def digest(d):
    h=hashlib.sha256()
    for s in sorted(glob.glob(d+'/*.safetensors')):
        with open(s,'rb') as f:
            for c in iter(lambda: f.read(1<<20), b''): h.update(c)
    return h.hexdigest()
import subprocess
lcver = subprocess.run(['./.venv/bin/pip','show','llmcompressor'],capture_output=True,text=True).stdout
lcver = next((l.split()[-1] for l in lcver.splitlines() if l.startswith('Version')), 'unknown')
for name, parent in (('qwen3-1.7b-int8-w8a8','models/parent_manifest.json'),
                     ('qwen3-8b-int8-w8a8','models/parent8b_manifest.json')):
    d='models/'+name
    cfg=json.load(open(d+'/config.json'))
    q=cfg.get('quantization_config') or cfg.get('compression_config')
    pm=json.load(open(parent))
    out={'checkpoint':name,'checkpoint_digest':digest(d),
         'parent_model':pm['model'],'parent_revision':pm['revision'],
         'quantization_config':q,
         'calibration_dataset':'open_platypus (llmcompressor builtin; revision unrecorded at quantization time — honest gap)',
         'generator':'llmcompressor '+lcver,
         'tokenizer_files_sha256':hashlib.sha256(''.join(
             hashlib.sha256(open(f,'rb').read()).hexdigest()
             for f in sorted(glob.glob(d+'/tokenizer*'))).encode()).hexdigest()}
    json.dump(out,open('checkpoint_manifest_v2_'+name+'.json','w'),indent=2)
    print('manifest', name, 'OK')
PYEOF
# 4) GPU: tf_v5 both arms (full identity manifest)
PREV=$(python3 -c "import json;print(json.load(open('models/parent_manifest.json'))['revision'])")
docker run --rm --gpus '"device=0"' --shm-size=2g -v $PWD:/w -w /w -v $PWD/models:/models:ro --entrypoint python3 $PIN /w/tf_v5.py --model /models/qwen3-1.7b-int8-w8a8 --rails /w/tokens64_CUTLASS.json --out /w/tf64v5_CUTLASS.json --n-prompts 64 --parent-revision $PREV
docker run --rm --gpus '"device=0"' --shm-size=2g -v $PWD:/w -w /w -v $PWD/models:/models:ro -e VLLM_DISABLED_KERNELS=CutlassScaledMMLinearKernel,CutlassInt8ScaledMMLinearKernel --entrypoint python3 $PIN /w/tf_v5.py --model /models/qwen3-1.7b-int8-w8a8 --rails /w/tokens64_CUTLASS.json --out /w/tf64v5_TRITON.json --n-prompts 64 --parent-revision $PREV
# 5) P4 v6(manifest join)
./.venv/bin/python3 w4_p4_stats_v4.py tf64v5_CUTLASS.json tf64v5_TRITON.json p4_stats64_v6_verified.json --isotonic
# 6) E2 first run (on failure, record it; do not change the code)
{
  echo "command: python3 /w/w5_first_divergence.py --model /models/qwen3-1.7b-int8-w8a8-pow2 --arm CUTLASS --prompt-idx 0 --out /w/fd_CUTLASS_0.json"
  echo "git_sha: 3dbd6bf"; echo "image: $PIN"
  docker run --rm --gpus '"device=0"' --shm-size=2g -v $PWD:/w -w /w -v $PWD/models:/models:ro --entrypoint python3 $PIN /w/w5_first_divergence.py --model /models/qwen3-1.7b-int8-w8a8-pow2 --arm CUTLASS --prompt-idx 0 --out /w/fd_CUTLASS_0.json
  echo "EXIT_e2=$?"
} > e2_first_run_record.txt 2>&1
echo EXEC_V6_DONE
