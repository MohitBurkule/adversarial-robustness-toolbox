"""Fix h219 SSL robustness script - replace broken attack_success calls."""
import re

path = 'hypotheses/h219_ssl_robustness_proxy.py'
with open(path, 'r') as f:
    content = f.read()

def attack_block(model_var):
    return f"""# Unfreeze for gradient-based attacks
    for p in {model_var}.parameters(): p.requires_grad_(True)
    X_fgsm = C.fgsm({model_var}, Xte, Yte, eps=EPS)
    X_pgd  = C.pgd({model_var},  Xte, Yte, eps=EPS, steps=10, alpha=0.01)
    with torch.no_grad():
        fgsm_asr = ({model_var}(X_fgsm).argmax(1) != Yte).float().mean().item()
        pgd_asr  = ({model_var}(X_pgd ).argmax(1) != Yte).float().mean().item()"""

# Fix all three model attack blocks
for model_var in ['model_sup', 'full_simclr', 'full_rot']:
    pattern = rf"fgsm_res = \{{.*?\}}\n    pgd_res  = C\.attack_success\({re.escape(model_var)}.*?\n"
    replacement = attack_block(model_var) + '\n'
    content = re.sub(pattern, replacement, content, count=1, flags=re.DOTALL)

# Fix any remaining fgsm_res/pgd_res references
content = content.replace("fgsm_res['asr']", "fgsm_asr")
content = content.replace("pgd_res['asr']",  "pgd_asr")

with open(path, 'w') as f:
    f.write(content)

print("Fixed h219. Lines with 'fgsm_asr':")
for i, line in enumerate(content.split('\n')):
    if 'fgsm_asr' in line or 'pgd_asr' in line:
        print(f"  {i+1}: {line.rstrip()}")
