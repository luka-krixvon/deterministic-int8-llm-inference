"""Regression tests for p5_checks. Each check must fire when it should and stay
silent when it should not; the boundaries are where these get written wrong."""
import torch, sys
from p5_checks import (Arm, run_suite, check1_shared_operands, check2_int32_no_overflow,
    check3_lossless_fp32_entry, check4_exact_accumulator, check5_pow2_scale_identity,
    check6_real_scale_tolerance, check7_token_level_risk, PRODUCT_BOUND_WITH_MIN_ACT,
    PRODUCT_BOUND_UNRESTRICTED, FP32_EXACT_INT_LIMIT)

P=F=0
def ck(name, cond):
    global P,F
    if cond: P+=1; print(f"PASS {name}")
    else:    F+=1; print(f"FAIL {name}")

def mk(name, M=4, N=3, K=8, pow2=True, out=None, acc=None, seed=0):
    g=torch.Generator().manual_seed(seed)
    a_q=torch.randint(-127,128,(M,K),generator=g,dtype=torch.int8)
    w_q=torch.randint(-127,128,(N,K),generator=g,dtype=torch.int8)
    s_a=(torch.full((M,1),0.25) if pow2 else torch.full((M,1),0.3))
    s_w=(torch.full((1,N),0.5)  if pow2 else torch.full((1,N),0.7))
    if out is None: out=torch.zeros(M,N,dtype=torch.bfloat16)
    return Arm(name,a_q,w_q,s_a,s_w,out,acc)

# --- check 1: bitwise, so signed zero must be caught
a=mk("a"); b=mk("b")
ck("c1 identical operands -> silent", not check1_shared_operands(a,b).fired)
b2=mk("b"); b2.s_a=b2.s_a.clone(); b2.s_a[0,0]=-0.0*1.0; b2.s_a[0,0]=torch.tensor(-0.0)
ck("c1 signed zero in scale -> fires", check1_shared_operands(mk("a"),b2).fired if True else False)
b3=mk("b",seed=1)
ck("c1 different operands -> fires", check1_shared_operands(a,b3).fired)

# --- check 2: the boundary. 16256*K < 2^31 -> K <= 132104
ck("c2 K=132104 admissible", not check2_int32_no_overflow(132104).fired)
ck("c2 K=132105 overflows",  check2_int32_no_overflow(132105).fired)
ck("c2 unrestricted K=131071 ok", not check2_int32_no_overflow(131071, PRODUCT_BOUND_UNRESTRICTED).fired)
ck("c2 unrestricted K=131072 overflows", check2_int32_no_overflow(131072, PRODUCT_BOUND_UNRESTRICTED).fired)
ck("c2 reports max_K", check2_int32_no_overflow(8).evidence["max_K_admissible"]==132104)

# --- check 3: 2^24 exactly is fine, 2^24+1 is not
ck("c3 exactly 2^24 -> silent", not check3_lossless_fp32_entry(torch.tensor([FP32_EXACT_INT_LIMIT])).fired)
ck("c3 2^24+1 -> fires", check3_lossless_fp32_entry(torch.tensor([FP32_EXACT_INT_LIMIT+1])).fired)
ck("c3 negative magnitude counted", check3_lossless_fp32_entry(torch.tensor([-(FP32_EXACT_INT_LIMIT+2)])).fired)

# --- check 4: not exposed -> inapplicable, not a pass
v=check4_exact_accumulator(mk("a"),mk("b"))
ck("c4 no accumulator -> inapplicable", (not v.applicable) and (not v.fired))
acc=torch.arange(12,dtype=torch.int32).reshape(4,3)
ck("c4 equal accs, matching ref -> silent",
   not check4_exact_accumulator(mk("a",acc=acc),mk("b",acc=acc.clone()),acc).fired)
ck("c4 arms differ -> fires",
   check4_exact_accumulator(mk("a",acc=acc),mk("b",acc=acc+1),acc).fired)
ck("c4 arms agree but ref differs -> fires",
   check4_exact_accumulator(mk("a",acc=acc),mk("b",acc=acc.clone()),acc+5).fired)

# --- check 5: pow2 gate, and bitwise (signed zero)
o1=torch.tensor([[1.0,2.0,4.0]]*4,dtype=torch.bfloat16)
ck("c5 pow2 scales + identical -> silent", not check5_pow2_scale_identity(mk("a",out=o1),mk("b",out=o1.clone())).fired)
o2=o1.clone(); o2[0,0]=1.0078125
ck("c5 pow2 scales + differing -> fires", check5_pow2_scale_identity(mk("a",out=o1),mk("b",out=o2)).fired)
v=check5_pow2_scale_identity(mk("a",pow2=False,out=o1),mk("b",pow2=False,out=o2))
ck("c5 non-pow2 scales -> inapplicable", (not v.applicable) and (not v.fired))
z1=torch.tensor([[0.0]],dtype=torch.bfloat16); z2=torch.tensor([[-0.0]],dtype=torch.bfloat16)
ck("c5 +0 vs -0 -> fires (bitwise)", check5_pow2_scale_identity(mk("a",M=1,N=1,out=z1),mk("b",M=1,N=1,out=z2)).fired)

# --- check 6: ULP distance, including across zero and the tolerance edge
x=torch.tensor([[1.0]],dtype=torch.bfloat16)
one_up=torch.tensor([[1.0078125]],dtype=torch.bfloat16)   # exactly 1 bf16 ulp above 1.0
two_up=torch.tensor([[1.015625]],dtype=torch.bfloat16)
v=check6_real_scale_tolerance(mk("a",M=1,N=1,out=x),mk("b",M=1,N=1,out=one_up))
ck("c6 1 ulp -> silent, reported as 1", (not v.fired) and v.evidence["max_ulp"]==1)
v=check6_real_scale_tolerance(mk("a",M=1,N=1,out=x),mk("b",M=1,N=1,out=two_up))
ck("c6 2 ulp -> fires", v.fired and v.evidence["max_ulp"]==2)
v=check6_real_scale_tolerance(mk("a",M=1,N=1,out=z1),mk("b",M=1,N=1,out=z2))
ck("c6 +0 vs -0 -> 0 ulp numerically but bitwise unequal (matches metrics_v3)",
   v.evidence["max_ulp"]==0 and v.evidence["bitwise_equal"] is False and not v.fired)
# the two mappings must agree by construction, not by coincidence
import torch as _t
def _v1_map(x):
    ia=x.detach().cpu().contiguous().view(_t.int16).to(_t.int32)
    return _t.where(ia>=0, ia+2**15, 2**15-(ia&0x7FFF)).to(_t.int64)
from p5_checks import _to_ordered_int as _p5_map
_probe=_t.tensor([[0.0,-0.0,1.0,-1.0,1.0078125,65504.0,-3.0e-5,6.1e-5]],dtype=_t.bfloat16)
ck("c6 mapping identical to metrics_v3 formula", bool(_t.equal(_p5_map(_probe), _v1_map(_probe))))
ck("c6 -1 to +1 spans 32512 steps",
   int((_p5_map(_t.tensor([[1.0]],dtype=_t.bfloat16))-_p5_map(_t.tensor([[-1.0]],dtype=_t.bfloat16))).abs().item())==32512)
nan=torch.tensor([[float('nan')]],dtype=torch.bfloat16)
v=check6_real_scale_tolerance(mk("a",M=1,N=1,out=nan),mk("b",M=1,N=1,out=nan.clone()))
ck("c6 all non-finite -> inapplicable", (not v.applicable) and (not v.fired))
ck("c6 threshold travels in verdict",
   check6_real_scale_tolerance(mk("a",M=1,N=1,out=x),mk("b",M=1,N=1,out=x),max_ulp=3).threshold==3.0)

# --- check 7: direction of the margin, ties, degenerate input
m=torch.tensor([0.1,0.2,5.0,6.0]); f=torch.tensor([True,True,False,False])
v=check7_token_level_risk(m,f,max_flip_rate=0.9)
ck("c7 small margins flip -> auc 1.0", abs(v.evidence["margin_auc"]-1.0)<1e-12)
v=check7_token_level_risk(m,~f,max_flip_rate=0.9)
ck("c7 reversed -> auc 0.0", abs(v.evidence["margin_auc"]-0.0)<1e-12)
v=check7_token_level_risk(torch.tensor([1.0,1.0]),torch.tensor([True,False]),max_flip_rate=0.9)
ck("c7 all ties -> auc 0.5", abs(v.evidence["margin_auc"]-0.5)<1e-12)
v=check7_token_level_risk(m,torch.tensor([False]*4),max_flip_rate=0.9)
ck("c7 no flips -> auc None, still applicable", v.applicable and v.evidence["margin_auc"] is None)
ck("c7 rate over tolerance -> fires", check7_token_level_risk(m,f,max_flip_rate=0.1).fired)
v=check7_token_level_risk(torch.tensor([]),torch.tensor([]))
ck("c7 empty -> inapplicable", not v.applicable)

# --- suite: inapplicable must not be counted as applicable
s=run_suite(mk("a"),mk("b"),K=8)
ck("suite counts 7 checks", s["_summary"]["n_checks"]==7)
ck("suite marks missing inputs inapplicable",
   set(s["_summary"]["inapplicable"])>={"lossless_fp32_entry","exact_accumulator","token_level_risk"})
ck("suite n_applicable < 7 when inputs missing", s["_summary"]["n_applicable"]<7)
s2=run_suite(mk("a",acc=acc),mk("b",acc=acc.clone()),K=8,acc_ref=acc,
             margins=m,flips=f,max_flip_rate=0.9)
ck("suite all applicable when fed", s2["_summary"]["n_applicable"]==7)
ck("suite clean case fires nothing", s2["_summary"]["n_fired"]==0)

print(f"\n{P} passed, {F} failed")
sys.exit(1 if F else 0)
