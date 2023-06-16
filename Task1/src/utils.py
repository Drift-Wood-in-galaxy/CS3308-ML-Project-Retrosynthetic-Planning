import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdchiral.template_extractor import extract_from_reaction as extracter

def Extract_smiles(reaction):
    reactants, products = str(reaction).split('>>')
    inputRec = {'_id' : None, 'reactants': reactants, 'products': products}
    ans = extracter(inputRec)
    if 'reaction_smarts' in ans.keys():
        return ans['reaction_smarts']
    else:
        return None
    
def Extract_products(reaction):
    reactants, products = str(reaction).split('>>')
    inputRec = {'_id' : None, 'reactants': reactants, 'products': products}
    ans = extracter(inputRec)
    if 'products' in ans.keys():
        return ans['products']
    else:
        return None
    
    
def Mole_to_vec(product):
    mol = Chem.MolFromSmiles(product)
    print(product)
    print(mol)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=2048)
    onbits = list(fp.GetOnBits())
    print(onbits)
    arr = np.zeros(fp.GetNumBits(), dtype =np.bool)
    arr[onbits] = 1
    return arr

