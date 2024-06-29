import torch
import sys

class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 2
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None, f=sys.stdout, key='', avr=-1):
        if run is not None:
            result = torch.tensor(self.results[run])
            argmax = result[:, 0].argmax().item()
            print(f'run: {run + 1},',
                  key,
                  f'best epoch={argmax + 1}',
                  f'val={result[:, 0].max():.4f}',
                  f'test: {result[argmax, 1]:.4f}'
                  )
        else:
            count = avr if avr != -1 else len(self.results)
            result = torch.tensor(self.results[:count])
            best_results = []
            for r in result:
                valid = r[:, 0].max().item()
                test = r[r[:, 0].argmax(), 1].item()
                best_results.append((valid, test))
            best_result = torch.tensor(best_results)

            r_v = best_result[:, 0]
            r_t = best_result[:, 1]
            print(key,
                  f'Average val={r_v.mean():.4f} ± {r_v.std():.4f}',
                  f'test={r_t.mean():.4f} ± {r_t.std():.4f}'
                  )

