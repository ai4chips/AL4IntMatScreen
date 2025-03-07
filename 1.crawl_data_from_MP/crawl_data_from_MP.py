
def Data_Generate(elements):
 
    import pandas as pd
    import itertools
    import re

    complx = list(itertools.combinations(elements,2))
    pool = []

    for k in complx :
        pool.append(k[0]+"-"+k[1])
    mp_id_lst = []

    api_key = "qFNhtxk39TYO5MjsA85v8HHiRcZy9E1S"

    from mp_api.client import MPRester

    m = MPRester(api_key)

    data = m.get_material_ids(pool)

    new_list = [re.search(r"mp-(\d+)", x).group(0) for x in data if re.search(r"mp-(\d+)", x)]

    df = pd.DataFrame(new_list, columns=['material_id'])

    return df


elements = ["Li","Be","Na","Mg","Al","K","Ca","Sc","Ti","V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge","Rb","Sr","Y","Zr", "Nb", "Mo", "Ru","Rh", "Pd", "Ag", "Cd", "In", "Sn","Sb","Cs","Ba","Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Tl", "Pb","Bi"]

data = Data_Generate(elements)

data.to_csv("1.crawl_data_from_MP/all_material_3_7_2025.csv",index=None)