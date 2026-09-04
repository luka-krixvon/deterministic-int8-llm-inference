import json
a=json.load(open("tokens_CUTLASS.json")); b=json.load(open("tokens_TRITON.json"))
per=[]
for x,y in zip(a,b):
    m=0
    for u,v in zip(x,y):
        if u!=v: break
        m+=1
    per.append((m,len(x)))
ident=sum(1 for m,n in per if m==n)
print(f"RESULT: {ident}/8 prompts bitwise-identical token sequences")
print("RESULT per-prompt matched/len:", per)
