from NAMmain import iter_main
import pandas as pd

if __name__ == '__main__':
    start = -0.9
    end = 0.9
    step = 0.1

    st = start
    a = []
    b = []
    c = []
    d = []
    e = []
    k = []
    
    print(start, end, step)
    while(st <= end):
        if round(st, 1) == 0:
            continue
        last_val, last_val_adv, best_val, best_val_adv, bound_rate = iter_main(round(st, 1))
        k.append(round(st,4))
        a.append(last_val)
        b.append(last_val_adv)
        c.append(best_val)
        d.append(best_val_adv)
        e.append(bound_rate)

        st += step
    df = pd.DataFrame({'k':k, 'last_val':a, 'last_val_adv':b, 'best_val':c, 'best_val_adv':d, 'bound_rate':e})
    
    df.to_csv(f"./k_logs/{start}_{end}_{step}.csv")
