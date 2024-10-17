import argparse
from .mcloud import *


def cli():
    parser = argparse.ArgumentParser(description="Downsample chemical pointclouds.")
    parser.add_argument('command', type=str, help="Specify if preprocessing, downsampling, or visualizing.  Should be 'mkcloud', 'dedense', 'vis', or 'mksheet'.")
    parser.add_argument('path_in', type=str, help="Path/file for input data.  Data should be a ndarray (pointcloud) if\n\
                                                   using 'dedense'/'vis' or text file with SMILES if using 'mkcloud'/'mksheet.")
    parser.add_argument("-o",'--path_out', type=str,default = 'def',  help="Path/file for output results.  Do not include extensions if making a chemical point cloud.")
    parser.add_argument("-f", "--fig",  action='store_true', default=False, help="Save chemical point cloud figure.")
    parser.add_argument("-d",'--down', type=str,default=None,  help="Path/file for downsample list.  Used with secondary commands 'vis' and 'mksheet'.")
    parser.add_argument("-a", "--alpha",  action='store_true', default=False, help="Use optimized alpha shapes for volume estimation.")
    parser.add_argument("-s", "--sep", type=str, default=',', help="Specify separator for SMILES file.")
    parser.add_argument("-p", "--pos", type=int, default=0, help="Specify position for SMILES in SMILES file. Note: position is zero indexed.")
    parser.add_argument("-r", "--rand", type=int, default=0, help="Random seed for downsampling.")
    parser.add_argument("-m", "--min", type=int, default=5, help="The min_size parameter for HDBSCAN.")
    parser.add_argument("-t", "--targ", type=float, default=0.5, help="Target downsampling percentage.")
    parser.add_argument("-e", "--epsilon", type=float, default=0.0, help="Cluster selection epsilon value for HDBSCAN.")
    parser.add_argument("-dw", "--dweight", type=float, default=None, help="Weighting term for density bias.")
    parser.add_argument("-vw", "--vweight", type=float, default=None, help="Weighting term for volume bias.")
    parser.add_argument("-c", "--cloud", type=str, default='chem_cloud.npy', help="Path/file for chemical point cloud (only when using 'mksheet' command).")
    parser.add_argument("-x", "--excel",  action='store_true', default=False, help="Use to load and save excel sheets rather than delimited text.  Default is False.")
    parser.add_argument("-H", "--header",  action='store_true', default=False, help="Specify if a header is present when loading sheets/delimited text.  Default is False.")
    parser.add_argument("-S", "--strict", action="store_true", default=False, help="Completely drops clusters with 'target values' of 0 rather than keeping a single molecule.")
    parser.add_argument("--SHOW", action="store_true", default=False, help="Display HDBSCAN clustering results prior to downsampling.")

    return parser.parse_args()


def main():
    valid = ['dedense','mkcloud','vis', 'mksheet']
    args = cli() # get user inputs from command line
    funct = args.command
    if funct not in valid:
        raise ValueError("Provide either 'dedense', 'mkcloud', 'vis' as function names.")
    if funct == 'vis' or funct == 'mksheet':
        down = args.down#Downsampled list
        path_in = args.path_in#SMILES
        fig_out = None
        if args.fig and funct == 'vis':
            fig_out = args.path_out
        elif args.fig and funct == 'mksheet':
            raise ValueError("Figure flag is not relevant for 'mksheet'.  Use '-o' or '--path_out' <$your_path>.")
        else:
            cloud = args.cloud
            path_out = args.path_out
            sep = args.sep
            pos = args.pos
        
    else:
        path_in = args.path_in
        path_out = args.path_out    
        sep = args.sep
        pos = args.pos
        rand = args.rand
        alpha = args.alpha
        targ = args.targ
        strict = args.strict
        d_weight = args.dweight
        v_weight = args.vweight
        epsilon = args.epsilon
        min_size = args.min
        show = args.SHOW


    if funct == 'dedense':
        if path_out == 'def':
            path_out = 'downsampled_chem_cloud'
        data = np.load(path_in)
        print('Loading dedenser...')
        from .dedenser import Dedenser
        #from .dedenser import Dedenser
        print('Dedensing...')
        dd = Dedenser(data,targ,rand,alpha,min_size,d_weight,v_weight,epsilon,strict,show)
        out_cloud = dd.downsample()
        np.save(path_out,out_cloud)
        print(f"Done! Saved dedensed index at: {path_out}.npy")

    elif funct == 'mkcloud':
        if path_out == 'def':
            path_out = 'chem_cloud'
        make_cloud(path_in, path_out, sep, pos, exl=args.excel,
                   heady=args.header)

    elif funct == 'vis':
        points = np.load(path_in)
        if down != None:
            points = points[np.load(down)]
        if args.path_out != 'def' and not args.fig:
            print(f"Did you wish to save '{args.path_out}.svg'?")
            print("Make sure to use '-f' or '--fig' flags to save figures if desired.")
        see_cloud(fig_out,points,args.fig)

    elif funct == 'mksheet':
        if path_out == 'def':
            path_out = 'dedensed_sheet'
        try:
            down = np.load(down)
        except:
            if down != None:
                raise ValueError("User must provide .npy file containing downsampled indexs.")
            else:
                raise ValueError(f"Could not load '{down}'.")
        save_cloud(smiles_loc=path_in, f_out=path_out, points=cloud,
                           sep=sep, position=pos, indx=down, exl=args.excel,
                           heady=args.header)


if __name__ == '__main__':
    main()
