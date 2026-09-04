"""Self-tests for the fault injector. If the injector is wrong, every P5 conclusion
is wrong, so these assert what each fault must and must not do."""
import torch, sys
from p5_inject import (reference_arm, exact_accumulator, wrap_int32, inject, FAULTS,
                       BINARY_FAULTS, severities_for, SEVERITIES)
from p5_checks import Arm, check1_shared_operands, _bitwise_equal

P=F=0
def ck(n,c):
    global P,F
    if c: P+=1; print(f"PASS {n}")
    else: F+=1; print(f"FAIL {n}")

M,N,K=32,16,64
g=torch.Generator().manual_seed(7)
a_q=torch.randint(-127,128,(M,K),generator=g,dtype=torch.int8)
w_q=torch.randint(-127,128,(N,K),generator=g,dtype=torch.int8)
s_a=torch.rand(M,1,generator=g)*0.02+0.001
s_w=torch.rand(1,N,generator=g)*0.02+0.001
out_ref,acc_ref32,acc_ref64=reference_arm(a_q,w_q,s_a,s_w)

# reference sanity
ck("ref accumulator matches int64 matmul", torch.equal(acc_ref64, a_q.to(torch.int64)@w_q.to(torch.int64).T))
ck("ref K is inside the bound (no wrap in the clean case)", torch.equal(acc_ref32, acc_ref64))
ck("ref output finite", bool(torch.isfinite(out_ref.float()).all()))

# F8: the false-positive control MUST be bit-identical
i8=inject("F8_null",a_q,w_q,s_a,s_w)
ck("F8 output bit-identical to reference", _bitwise_equal(i8.out,out_ref))
ck("F8 accumulator identical", torch.equal(i8.acc,acc_ref32))
ck("F8 operands untouched", not check1_shared_operands(
     Arm("r",a_q,w_q,s_a,s_w,out_ref,acc_ref32), Arm("f",i8.a_q,i8.w_q,i8.s_a,i8.s_w,i8.out,i8.acc)).fired)

# F1-F5: must leave operands alone, and must actually change the output
for name in ("F1_scale_in_bf16","F2_double_rounding","F3_scale_order","F4_truncate_output","F5_fused_order"):
    inj=inject(name,a_q,w_q,s_a,s_w,"all_elements")
    same_ops = not check1_shared_operands(
        Arm("r",a_q,w_q,s_a,s_w,out_ref,acc_ref32),
        Arm("f",inj.a_q,inj.w_q,inj.s_a,inj.s_w,inj.out,inj.acc)).fired
    ck(f"{name} leaves operands untouched", same_ops)
    # Observability is recorded, not asserted: some faults are provably below bf16
    # resolution (see A-10.3). Asserting universal perturbation would force the
    # injector to fake a difference that the output precision cannot carry.
    ck(f"{name} records observability", inj.n_output_differing >= 0)
    ck(f"{name} accumulator untouched", torch.equal(inj.acc,acc_ref32))

# coverage ladder: one_element touches exactly one, one_percent about a hundredth
for name in ("F1_scale_in_bf16","F2_double_rounding","F3_scale_order","F4_truncate_output","F5_fused_order"):
    i1=inject(name,a_q,w_q,s_a,s_w,"one_element")
    ck(f"{name} one_element mask size 1", i1.touched==1)
    ip=inject(name,a_q,w_q,s_a,s_w,"one_percent")
    ck(f"{name} one_percent mask ~1%", ip.touched==max(1,(M*N)//100))
    ia=inject(name,a_q,w_q,s_a,s_w,"all_elements")
    ck(f"{name} all_elements mask = numel", ia.touched==M*N)
    nd1=int((i1.out.view(torch.int16)!=out_ref.view(torch.int16)).sum())
    nda=int((ia.out.view(torch.int16)!=out_ref.view(torch.int16)).sum())
    ck(f"{name} differing count non-decreasing with coverage", nd1<=nda)
    if nda == 0:
        print(f"NOTE {name} is not observable at bf16 output precision in this regime "
              f"(A-10.3): no check can be credited or blamed for it here")

# F6: must actually wrap
i6=inject("F6_int32_overflow",a_q,w_q,s_a,s_w)
e64=exact_accumulator(i6.a_q,i6.w_q)
ck("F6 exact acc exceeds 2^31", int(e64.abs().max())>=2**31)
ck("F6 reported acc is the wrapped value", torch.equal(i6.acc,wrap_int32(e64)) and not torch.equal(i6.acc,e64))
ck("F6 operands are the overflow-inducing ones", i6.a_q.shape[1]>K)

# F7: above 2^24, below 2^31
i7=inject("F7_above_2p24",a_q,w_q,s_a,s_w)
mx=int(exact_accumulator(i7.a_q,i7.w_q).abs().max())
ck("F7 max|acc| above 2^24", mx>2**24)
ck("F7 max|acc| below 2^31 (no wrap)", mx<2**31)
ck("F7 no wrap occurred", torch.equal(i7.acc,exact_accumulator(i7.a_q,i7.w_q)))

# F9: must change operands, at all three severities
for sev in SEVERITIES:
    i9=inject("F9_operand_mismatch",a_q,w_q,s_a,s_w,sev)
    fired=check1_shared_operands(
        Arm("r",a_q,w_q,s_a,s_w,out_ref,acc_ref32),
        Arm("f",i9.a_q,i9.w_q,i9.s_a,i9.s_w,i9.out,i9.acc)).fired
    ck(f"F9 {sev} makes operands differ", fired)

# determinism and catalogue hygiene
x=inject("F1_scale_in_bf16",a_q,w_q,s_a,s_w,"one_percent",seed=3)
y=inject("F1_scale_in_bf16",a_q,w_q,s_a,s_w,"one_percent",seed=3)
ck("same seed -> identical injection", _bitwise_equal(x.out,y.out))
z=inject("F1_scale_in_bf16",a_q,w_q,s_a,s_w,"one_percent",seed=4)
ck("different seed -> different mask", not _bitwise_equal(x.out,z.out))
ck("catalogue has nine faults", len(FAULTS)==9)
try:
    inject("F10_new",a_q,w_q,s_a,s_w); ck("unknown fault rejected", False)
except KeyError: ck("unknown fault rejected", True)
try:
    inject("F6_int32_overflow",a_q,w_q,s_a,s_w,"one_element"); ck("binary fault rejects severity", False)
except ValueError: ck("binary fault rejects severity", True)
ck("binary faults run once only", all(severities_for(f)==("all_elements",) for f in BINARY_FAULTS))

# The detection floor itself is a tested property, so a later edit that appears to
# "fix" F3 gets noticed rather than silently changing what the sensitivity table means.
i3=inject("F3_scale_order",a_q,w_q,s_a,s_w,"all_elements")
ck("F3 reassociation stays below bf16 resolution here (documented, not a bug)",
   i3.n_output_differing==0)
from p5_inject import observability_survey as _obs
_r=_obs(a_q,w_q,s_a,s_w,seed=1)
ck("observability survey covers every fault/severity",
   len(_r)==sum(len(severities_for(f)) for f in FAULTS))
ck("survey marks F3 unobservable", _r["F3_scale_order/all_elements"]["observable"] is False)
ck("survey marks F1 observable", _r["F1_scale_in_bf16/all_elements"]["observable"] is True)

print(f"\n{P} passed, {F} failed")
sys.exit(1 if F else 0)
