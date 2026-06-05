"""Fix remaining stale fgsm_res/pgd_res dict references in h219."""
path = 'hypotheses/h219_ssl_robustness_proxy.py'
with open(path, 'r') as f:
    content = f.read()

content = content.replace('fgsm_res["asr"]', 'fgsm_asr')
content = content.replace("fgsm_res['asr']", 'fgsm_asr')
content = content.replace('pgd_res["asr"]',  'pgd_asr')
content = content.replace("pgd_res['asr']",  'pgd_asr')

with open(path, 'w') as f:
    f.write(content)
print("Done. Remaining fgsm_res refs:", content.count("fgsm_res"), "pgd_res refs:", content.count("pgd_res"))
