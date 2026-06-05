"""Remove H406b checkpoint / DONE / output so a smoke test starts fresh."""
import os

D = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "results", "fashion_mnist")
for f in ["h406b_weekend.ckpt", "h406b_weekend.ckpt.tmp", "h406b_weekend.DONE",
          "h406b_weekend_maxscale_output.txt", "h406b_weekend_maxscale_output.txt.tmp",
          "h406b_weekend.log"]:
    p = os.path.join(D, f)
    if os.path.exists(p):
        os.remove(p)
        print("removed", p)
    else:
        print("absent ", p)
print("reset done")
