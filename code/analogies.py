
def read_google_analogies(filename):
    """ Parse file and return a list of dicts.
    Google's analogy data has 4 space separated words per line,
    except for lines starting with ":", which indicate types of analogy.
    """
    analogy_dicts = []
    analogy_types = []
    with open(filename) as f:
        for line in f:
            if line.startswith(":"):
                analogy_types.append(line[2:-1])
                analogy_dicts.append({})
            else:
                w1,w2,w3,w4 = line.split()
                d0 = analogy_dicts[-1]
                d1 = d0.get(w1, {})
                d2 = d1.get(w2, {})
                d2[w3] = w4
                d1[w2] = d2
                d0[w1] = d1
        return analogy_dicts, analogy_types


