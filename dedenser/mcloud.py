"""Utils for making chemical point clouds, visualization, and saving results.
"""

import pandas as pd
import numpy as np

def drop_non_numeric_columns(df):
    """
    Drop non-numeric columns from a DataFrame.

    Arguments
    ---------
    df : pd.DataFrame 
        The input DataFrame.

    Returns
    -------
    df_out : pd.DataFrame
        The modified DataFrame with non-numeric columns dropped.
    """
    non_numeric_columns = []
    # Iterate over each column in the DataFrame
    for column in df.columns:
        # Check if the column contains any non-numeric values
        if pd.to_numeric(df[column], errors='coerce').isnull().any():
            non_numeric_columns.append(column)
    # Drop the non-numeric columns from the DataFrame
    df_out = df.drop(columns=non_numeric_columns)
    return df_out


def make_cloud(path, path_out='cloud_out', sep=',', position=0, exl=False, heady=False):
    """
    Featurize SMILES strings, perform dimensionality reduction using UMAP and save
    the resulting chemical point cloud.  Mordred descriptors that result in errors
    or otherwise non-numeric values are dropped with :func:`drop_non_numeric_columns`

    Parameters
    ----------
    path : str
        The path to the file containing SMILES strings.  Should include file extension.

    path_out : str, default='cloud_out'
        The output file name for the resulting chemical point cloud. Default is 'cloud_out'.
        Morded descriptors are also saved as a CSV using the same file name.

    sep : str, default=','
        The separator used in the file containing SMILES strings. Default is ','.

    position : int, default=0
        The index of the column containing the SMILES strings. Default is 0.

    exl : bool, default=False 
        Flag indicating whether the input file is an Excel sheet.

    heady : bool, default=False
        Indicates if there is a header in the input SMILES file.

    Returns
    -------
    cloud : numpy.ndarray of shape (N, 3)
        Coordinates  for the resulting chemical point cloud (UMAP projection). 
    """
    print('Loading Scikit-learn, RDKit, and Mordred...')
    from mordred import Calculator, descriptors
    from rdkit import Chem
    import sklearn
    print('Finished loading dependencies, featurizing SMILES...')
    scaler = sklearn.preprocessing.StandardScaler()
    calc = Calculator(descriptors, ignore_3D=True)
    print('Loading SMILES...')
    if heady:
        head = 0
    else:
        head = None
    if exl:
        smiles = pd.read_excel(path,header=head)
    else:
        smiles = pd.read_csv(path,sep=sep,header=head,engine='python')
    smiles = smiles.to_numpy()[:,position]
    print('Converting to Mols...')
    mols = [Chem.MolFromSmiles(smi) for smi in smiles]
    print('Calculating 2D descriptors from Mols...')
    df = calc.pandas(mols)
    df = drop_non_numeric_columns(df)
    new_column = pd.Series(smiles, name='SMILES')
    s_vals = scaler.fit_transform(df.values)
    print("Finished 2D descriptor calculations.")
    print("Loading UMAP and embedding chemical point cloud...")
    from umap import UMAP
    reducer = UMAP(n_components=3, min_dist=.1, spread=.5, n_neighbors=15)
    cloud = reducer.fit_transform(s_vals)
    np.save(path_out,cloud)
    df.insert(0, 'SMILES', new_column)
    df.to_csv(f'{path_out}.csv',index=False)
    print(f"Done! Saved chemical point cloud at '{path_out}.npy'.")
    print(f"Saved 2D descriptors at '{path_out}.csv'.")

def see_cloud(f_out, points, save=False):
    """
    Visualize the chemical point cloud in a 3D scatter plot.

    Parameters
    ----------
    f_out (str): The output file name or path (excluding file extension) for the plot.
        
    points : numpy.ndarray of shape (N, 3)
        The array of points representing the chemical point cloud.
        
    save : bool, default=False
        Flag indicating whether to save the plot. Default is False.
    """
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2],s=.1,c='b')
    ax.set_xlabel('UMAP_1')
    ax.set_ylabel("UMAP_2")
    ax.set_zlabel("UMAP_3")
    plt.tight_layout()
    if save:
        plt.savefig(f'{f_out}.svg')
    plt.show()

def save_cloud(smiles_loc, f_out, points='chem_cloud.npy', sep=',',
                       position=0, indx=None, exl=False, heady=False):
    """
    Save the chemical point cloud with corresponding SMILES strings to
    a human readable file.

    Parameters
    ----------
    smiles_loc : str
        The path/name of the file containg the SMILES to be saved.

    f_out : str
        The path/name of the file to be saved.

    points : str, default='chem_cloud.npy'
        The file path/name for the chemical point cloud.

    sep : str, default=','
        The delimitor to be used when parsing the file with SMILES.
    
    position : int, default=0
        The index location to be used when reading SMILES.
    
    indx : str or None, default=None
        The .npy file path/name of the downsampling indexs.  If the
        '-d' or '--down' flags were not used for the comand line, or
        if otherwise left as None, then the full point cloud is saved.

    exl : bool, default=False
        Flag indicating wether the input file is an Excel sheet. If set
        to True, the resulting output file will aslo be an Excel sheet.
    
    heady : bool, default=False
        Flag to indicate if the SMILES file contains a header line or not.

    Returns
    -------
    f_out : pd.DataFrame of shape (points, 4)
        Dataframe to be saved containing SMILES and 3-D UMAP embeddings of
        a chemical point cloud.
    """
    if heady:
        head = 0
    else:
        head = None
    if exl:
        try:
            smiles = pd.read_excel(smiles_loc,header=head)
        except Exception as error:
            raise ValueError(f"Issue loading '{smiles_loc}'.  \
                             If using the excel flag, make sure\
                             you are also loading excel sheets.") from error
    else:
        smiles = pd.read_csv(smiles_loc,sep=sep,header=head,engine='python')
    smiles = smiles.to_numpy()[:,position]
    points = np.load(points)
    if indx is not None:
        smiles = smiles[indx]
        points = points[indx]
    sheet = pd.DataFrame({'SMILES': smiles,
                          'UMAP_1': points[:,0],
                          'UMAP_2': points[:,1],
                          'UMAP_3': points[:,2]
                        })
    if exl:
        sheet.to_excel(f_out,index=False)
    else:
        sheet.to_csv(f_out,index=False)
    print(f'Completed with no errors, wrote results to {f_out}')
