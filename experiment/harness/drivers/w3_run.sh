#!/bin/bash
cd ~/integer_alibi
exec > logs/w3_ksweep.log 2>&1
PIN=vllm/vllm-openai@sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967
docker run --rm --gpus '"device=0"' --shm-size=2g -v $PWD:/w -w /w --entrypoint python3 $PIN /w/w3_ksweep.py
echo W3_DONE
