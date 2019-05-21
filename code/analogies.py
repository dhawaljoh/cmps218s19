
DATA_DIR = "../data/"

def parse_google_analogies(filename, type=None):
    """ Parse file and return a list of dicts.
    Google's analogy data has 4 space separated words per line,
    except for lines starting with ":", which indicate types of analogy.
    """
    analogy_dicts = []
    analogy_tupls = []
    analogy_types = []
    with open(DATA_DIR + filename) as f:
        skip_type = False
        for line in f:
            if line.startswith(":"):
                curr_type = line[2:-1]
                if type:
                    if type != curr_type:
                        skip_type = True
                        continue
                    else:
                        skip_type = False
                analogy_types.append(curr_type)
                analogy_tupls.append([])
                analogy_dicts.append({})

            else:
                if type:
                    if skip_type:
                        continue
                    else:
                        pass
                w1,w2,w3,w4 = line.split()
                analogy_tupls[-1].append((w1, w2, w3, w4))
                d0 = analogy_dicts[-1]
                d1 = d0.get(w1, {})
                d2 = d1.get(w2, {})
                d2[w3] = w4
                d1[w2] = d2
                d0[w1] = d1
        return analogy_dicts, analogy_tupls, analogy_types


