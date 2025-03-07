def P_Generate(material_id):
    
    import pandas as pd

    api_key = "qFNhtxk39TYO5MjsA85v8HHiRcZy9E1S"

    import warnings

    # 忽略警告
    warnings.filterwarnings("ignore", category=FutureWarning)

    from mp_api.client import MPRester
    with MPRester(api_key) as mpr:
        docs =  mpr.materials.summary.search(material_ids= material_id)
        data_list = []
        for doc in docs:
            data_list.append([doc.material_id, doc.formula_pretty,doc.band_gap, doc.bulk_modulus, doc.cbm, doc.composition, doc.density, doc.density_atomic, doc.dos_energy_down, doc.dos_energy_up, doc.e_electronic, doc.e_ij_max, doc.e_ionic, doc.e_total, doc.efermi, doc.energy_above_hull, doc.energy_per_atom, doc.equilibrium_reaction_energy_per_atom, doc.es_source_calc_id, doc.formation_energy_per_atom, doc.grain_boundaries, doc.has_reconstructed, doc.homogeneous_poisson, doc.is_gap_direct, doc.is_magnetic, doc.is_metal, doc.is_stable, doc.n, doc.nelements, doc.nsites, doc.num_magnetic_sites, doc.num_unique_magnetic_sites, doc.ordering, doc.shape_factor, doc.shear_modulus, doc.surface_anisotropy, doc.symmetry, doc.task_ids, doc.theoretical, doc.total_magnetization, doc.total_magnetization_normalized_formula_units, doc.total_magnetization_normalized_vol, doc.uncorrected_energy_per_atom, doc.universal_anisotropy, doc.vbm, doc.volume, doc.weighted_surface_energy, doc.weighted_surface_energy_EV_PER_ANG2, doc.weighted_work_function])
        Data = pd.DataFrame(data_list, columns=['material_id', 'formula_pretty','band_gap', 'bulk_modulus', 'cbm', 'composition', 'density', 'density_atomic', 'dos_energy_down', 'dos_energy_up', 'e_electronic', 'e_ij_max', 'e_ionic', 'e_total', 'efermi', 'energy_above_hull', 'energy_per_atom', 'equilibrium_reaction_energy_per_atom', 'es_source_calc_id', 'formation_energy_per_atom', 'grain_boundaries', 'has_reconstructed', 'homogeneous_poisson', 'is_gap_direct', 'is_magnetic', 'is_metal', 'is_stable', 'n', 'nelements', 'nsites', 'num_magnetic_sites', 'num_unique_magnetic_sites', 'ordering', 'shape_factor', 'shear_modulus', 'surface_anisotropy', 'symmetry', 'task_ids', 'theoretical', 'total_magnetization', 'total_magnetization_normalized_formula_units', 'total_magnetization_normalized_vol', 'uncorrected_energy_per_atom', 'universal_anisotropy', 'vbm', 'volume', 'weighted_surface_energy', 'weighted_surface_energy_EV_PER_ANG2', 'weighted_work_function'])

    return Data

def X_Generate(P_Descriptor):
    import pandas as pd

    from xenonpy.descriptor import Compositions
    from xenonpy.descriptor import Structures
    import re
    from mp_api.client import mprester

    from xenonpy.datatools import preset

    preset.sync('elements')
    preset.sync('elements_completed')

    lst_Com= P_Descriptor['composition']

    lst_Com_Dic=[]

    for string in lst_Com:

        split_string = string.split()
        result_dict = {}

        for item in split_string:
            element = ''.join(filter(str.isalpha, item))
            number = int(''.join(filter(str.isdigit, item)))
            result_dict[element] = number
        lst_Com_Dic.append(result_dict)
        
    MP_API_KEY = "5vDJu0MvvXZtFQ4T3a0d8AU7ZQzh4aFD"

    with mprester.MPRester(MP_API_KEY) as mpr:
        Com = Compositions()
        descriptor_com = Com.transform(lst_Com_Dic)
        descriptor_com.insert(0, 'formula_pretty', P_Descriptor['formula_pretty'])
        descriptor_com.insert(0, 'material_id', P_Descriptor['material_id'])


    id_lst = P_Descriptor["material_id"].tolist()

    Str_id = []
    material_id=[]
    MP_API_KEY = "5vDJu0MvvXZtFQ4T3a0d8AU7ZQzh4aFD"
    with mprester.MPRester(MP_API_KEY) as mpr:
        docs =  mpr.materials.summary.search(material_ids= id_lst)
        for doc in docs:
            material_id.append(doc.material_id)
            Str_id.append(doc.structure)
        material_id = [re.search(r"mp-(\d+)", x).group(0) for x in material_id if re.search(r"mp-(\d+)", x)]

    print("Str_lst长度为"+str(len(Str_id)))

    # print(Str_id[0])
    Str = Structures()
    descriptor_Str = Str.transform(Str_id)

 
    new_df = pd.DataFrame({"material_id": material_id})

    descriptor_Str = pd.concat([new_df, descriptor_Str], axis=1)

    descriptor_Str = descriptor_Str.iloc[descriptor_Str["material_id"].apply(lambda x: id_lst.index(x)).argsort()]

    descriptor_Str = descriptor_Str.reset_index(drop=True)

    descriptor_Str.insert(1, "formula_pretty", P_Descriptor["formula_pretty"])

    return descriptor_com, descriptor_Str



import pandas as pd
df = pd.read_csv(".//Input//material_id.csv")
material_id = list(df["material_id"])
Data  = P_Generate(material_id)

Data.to_csv('.//Input//P_Descriptor.csv',index=False)

#环境为X
import pandas as pd

P_Descriptor = pd.read_csv(".//Output//P_Descriptor.csv")

X_Com , X_Str = X_Generate(P_Descriptor)

X_Com.to_csv(".//Output//X_Com_Descriptor.csv", encoding='utf_8_sig',index=False)
X_Str.to_csv(".//Output//X_Str_Descriptor.csv", encoding='utf_8_sig',index=False)