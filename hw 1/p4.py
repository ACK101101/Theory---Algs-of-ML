import numpy as np

def entropy(X):
    total = 0
    for x in X:
        if x != 0:
            total += (-1 * x * np.log2(x))
    return total

def infoGain(before, after):
    e_b = entropy(before)           # entropy before   
    print("Before: ", e_b)
    e_a = 0                         # entropy after
    for x in after:
        print("term: ", entropy(x))
        e_a += (0.5) * entropy(x)
    print("After: ", e_a)
    return e_b - e_a

def main():
    before = [5/8, 3/8]
    after = [[1/2, 1/2], [3/4, 1/4]]                        # A: [[1, 0], [1/4, 3/4]]
    gain = infoGain(before, after)
    print(gain)
    return

if __name__ == "__main__":
    main()