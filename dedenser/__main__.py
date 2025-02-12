import argparse
from .mcloud import *
import subprocess
from dash import Dash, dcc, html, Input, Output, callback


def cli():
    parser = argparse.ArgumentParser(description="Downsample chemical pointclouds.")
    parser.add_argument('command', type=str, help="Specify if preprocessing, downsampling, or visualizing.  Should be 'mkcloud', 'dedense', 'vis', 'mksheet', or 'Dash-app'.")
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
    parser.add_argument("-c", "--cloud", type=str, default='chem_cloud.npy', help="Path/file for chemical point cloud (only when using 'mksheet' or Dash-app command).")
    parser.add_argument("-x", "--excel",  action='store_true', default=False, help="Use to load and save excel sheets rather than delimited text.  Default is False.")
    parser.add_argument("-H", "--header",  action='store_true', default=False, help="Specify if a header is present when loading sheets/delimited text.  Default is False.")
    parser.add_argument("-S", "--strict", action="store_true", default=False, help="Completely drops clusters with 'target values' of 0 rather than keeping a single molecule.")
    parser.add_argument("--SHOW", action="store_true", default=False, help="Display HDBSCAN clustering results prior to downsampling.")

    return parser.parse_args()


def main():
    valid = ['dedense','mkcloud','vis', 'mksheet', 'Dash-app']
    args = cli() # get user inputs from command line
    funct = args.command
    if funct not in valid:
        raise ValueError("Provide either 'dedense', 'mkcloud', 'vis', 'Dash-app' as function names.")
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
        if path_in.split('.')[-1] == 'csv':
            data = pd.read_csv(path_in,sep=sep)
            data = data.values[:,1:]
        else:
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
    
    elif funct == 'Dash-app':
        from dash import Dash, dcc, html, Input, Output, callback
        import rdkit
        import plotly.express as px
        import numpy as np
        import json
        import pandas as pd
        import base64
        from io import BytesIO
        import sklearn 
        from rdkit import Chem
        from rdkit.Chem.rdchem import Mol
        from rdkit.Chem.Draw import rdMolDraw2D
        import argparse
        from .dedenser import Dedenser
        
        external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

        app = Dash(__name__, external_stylesheets=external_stylesheets)

        styles = {
            'pre': {
                'border': 'thin lightgrey solid',
                'overflowX': 'scroll'
            }
        }

        smiles = pd.read_csv(path_in)['SMILES']
        points = np.load(args.cloud)
        df = pd.DataFrame({'SMILES': smiles, 'UMAP_1': points[:,0], 'UMAP_2': points[:,1], 'UMAP_3': points[:,2]})

        fig = px.scatter_3d(df, x="UMAP_1", y="UMAP_2", z="UMAP_3", custom_data=["SMILES"])

        fig.update_layout(clickmode='event+select',plot_bgcolor='#f1f6f4',paper_bgcolor='#f1f6f4')

        fig.update_traces(marker_size=2,marker=dict(color='grey'))
        app.layout = html.Div([
            html.Div([
                dcc.Graph(
                    id='basic-interactions',
                    figure=fig,
                    style={'width': '67%', 'height': '70vh', 'display': 'inline-block','padding': '2px','backgroundColor': '#D3D3D3',}
                ),
                html.Div([
                    html.Div([
                        html.Img(
                            id='rdkit-display',
                            src='cccccc',
                            alt='image',
                            style={'width': '100%', 'height': '100%',},
                            **{'data-format': 'svg'}
                        )
                    ], style={'width': '90%', 'height': '50%', 'margin': 'auto', 'overflow': 'hidden'}),
                    html.Div([
                        dcc.Markdown("""
                            **Click Data**
                    
                            Click on points in the graph to visualize molecules.
                        """),
                        html.Pre(id='click-data', style=styles['pre'], className='click-data')
                    ], style={'width': '90%', 'height': '50%', 'margin': 'auto', 'overflow': 'auto'})
                ], style={'width': '30%', 'height': '70vh', 'display': 'inline-block', 'vertical-align': 'top'})
            ], style={'width': '100%', 'height': '70vh','padding': '2px','backgroundColor': 'white'}),

            # New row for Strict radio options
            html.Div([
                html.Div([
                    dcc.RadioItems(
                        id='strict-options',
                        options=[
                            {'label': 'Default', 'value': 'STRICT1'},
                            {'label': 'Strict', 'value': 'STRICT2'},
                        ],
                        value='STRICT1',  # Default selected value
                        labelStyle={'display': 'inline-block', 'margin': '10px'}  # Inline display and margin
                    ),
                ], style={'textAlign': 'center', 'marginTop': '10px'}),  # Reduced marginTop for radio options
                
            ], style={'width': '100%', 'textAlign': 'center'}),  # Full width for strict options row

            # Row for inputs and weighting options
            html.Div(style={'display': 'flex', 'justifyContent': 'center', 'marginTop': '10px', 'width': '100%'}, children=[
                html.Div(style={'display': 'flex', 'alignItems': 'center'}, children=[
                    html.Label('Target:', style={'fontWeight': 'bold', 'marginRight': '5px'}),
                    dcc.Input(id='d-input', placeholder='(0-1)', style={'margin': '5px','width': '75px'}),
                    
                    html.Label('Min. Size:', style={'fontWeight': 'bold', 'marginRight': '5px'}),
                    dcc.Input(id='m-input', value='5', style={'margin': '5px', 'width': '90px'}),
                    
                    html.Label('Epsilon:', style={'fontWeight': 'bold', 'marginRight': '5px'}),
                    dcc.Input(id='e-input', value='0.0', style={'margin': '5px', 'width': '90px'}),
                    
                    html.Button('Update Plot', id='update-button', n_clicks=0, style={'marginLeft': '10px','backgroundColor': 'white'}),
                ])
            ]),  # Centered inputs

            html.Div([
                html.Label('Weighting Option:', style={'fontWeight': 'bold'}),
                dcc.RadioItems(
                    id='w-options',
                    options=[
                        {'label': 'No Weight', 'value': 'OPT1'},
                        {'label': 'Volume', 'value': 'OPT2'},
                        {'label': 'Density', 'value': 'OPT3'},
                    ],
                    value='OPT1',  # Default selected value
                    labelStyle={'display': 'inline-block', 'margin': '10px'}  # Inline display and margin
                ),
                dcc.Input(id='weight', placeholder='Weight Value', style={'margin': '5px'}),
            ], style={'display': 'flex', 'justifyContent': 'center', 'marginTop': '10px', 'width': '100%'})  # Adjusted marginTop

        ], style={'backgroundColor': '#d5e4dd','width': '100%', 'height': '100vh'})
            

        @callback(
            Output('click-data', 'children'),
            Input('basic-interactions', 'clickData'))
        def display_click_data(clickData):
            if clickData is None:
                return ''
            
            point_info = clickData['points'][0] 

            # Extract UMAP coordinates
            umap_1 = point_info['x']
            umap_2 = point_info['y']
            umap_3 = point_info['z']

            # Extract custom data
            custom_data = point_info['customdata']
            smiles = custom_data[0]
            if len(custom_data) == 2:  # First element in custom data
                cluster_value = custom_data[1]  # Second element in custom data
            else:
                cluster_value = 'NA'

            # Create formatted output
            formatted_output = [
                f"UMAP_1: {umap_1}",
                f"UMAP_2: {umap_2}",
                f"UMAP_3: {umap_3}",
                f"SMILES: '{smiles}'",
                f"Cluster: '{cluster_value}'"
            ]
            return html.Pre("\n".join(formatted_output))

        def create_mol_plot(smiles):
                smiles = smiles
                buffered = BytesIO()
                if isinstance(smiles, str):
                    # Generate 2D SVG if smiles column is a string
                    svg_width, svg_height = 1000, 1000
                    d2d = rdMolDraw2D.MolDraw2DSVG(svg_width, svg_height)
                    opts = d2d.drawOptions()
                    opts.clearBackground = False
                    d2d.DrawMolecule(Chem.MolFromSmiles(smiles))
                    d2d.FinishDrawing()
                    img_str = d2d.GetDrawingText()
                    buffered.write(str.encode(img_str))
                    img_str = base64.b64encode(buffered.getvalue()).decode('ascii')
                    img_str = f"data:image/svg+xml;base64,{img_str}"
                    return img_str

        @app.callback(
            Output('basic-interactions', 'figure'),
            Output('update-button', 'n_clicks'),
            Input('update-button', 'n_clicks'),
            Input('d-input', 'value'),  
            Input('m-input', 'value'),  
            Input('e-input', 'value'),
            Input('strict-options', 'value'),
            Input('w-options', 'value'), 
            Input('weight', 'value'), 
            Input('basic-interactions', 'figure'),
            prevent_initial_call=True,    
        )
        def update_plot(n_clicks, t_value, m_value, e_value, s_value, w_bool, w_val, figure):
            # Validate that all input values are provided
            if not n_clicks is None:
                if n_clicks > 0 and None not in (t_value, m_value, e_value):
                    d_weight,v_weight = None, None
                    if s_value == 'STRICT2':
                        strict = True
                    else:
                        strict = False

                    if not w_bool == 'OPT1':
                        if not w_val is None:
                            if w_bool == 'OPT2':
                                v_weight = float(w_val)
                            else:
                                d_weight = float(w_val)
                    
                    df = pd.DataFrame({'SMILES': smiles, 'UMAP_1': points[:,0], 'UMAP_2': points[:,1], 'UMAP_3': points[:,2]})
                    dd = Dedenser(points,float(t_value),1,False,int(m_value),d_weight,v_weight,float(e_value),strict,False,True)
                    out_cloud, color = dd.downsample()
                    print('Plotting...')
                    the_colors = generate_colors(color)
                    df = pd.DataFrame({'SMILES': smiles, 'UMAP_1': points[:,0], 'UMAP_2': points[:,1],'cvals': the_colors, 'UMAP_3': points[:,2],'clusters': color})
                    dd = Dedenser(points,float(t_value),1,False,int(m_value),d_weight,v_weight,float(e_value),strict,False,True)
                    new_df = df.iloc[out_cloud]
                    new_fig = px.scatter_3d(new_df, x="UMAP_1", y="UMAP_2", z="UMAP_3", custom_data=["SMILES",'clusters'])
                    new_fig.update_layout(clickmode='event+select',plot_bgcolor='#f1f6f4',paper_bgcolor='#f1f6f4')
                    new_fig.update_traces(marker_size=4,marker=dict(color=new_df['cvals']))
                    print('Done')
                    return new_fig, None
                elif n_clicks > 0:
                    print('Target value must be provided.')
                    df = pd.DataFrame({'SMILES': smiles, 'UMAP_1': points[:,0], 'UMAP_2': points[:,1], 'UMAP_3': points[:,2]})

                    fig = px.scatter_3d(df, x="UMAP_1", y="UMAP_2", z="UMAP_3", custom_data=["SMILES"])

                    fig.update_layout(clickmode='event+select')

                    fig.update_traces(marker_size=2,marker=dict(color='grey'))
                    return fig, None
                return figure, None
            else:
                return figure, None
                
        @app.callback(
            Output('rdkit-display', 'src'),
            Input('basic-interactions', 'clickData')
        )
        def update_mol(selectedData):
            if selectedData is not None:
                smile_string = selectedData['points'][0]['customdata'][0]
                return create_mol_plot(smile_string)
            
        app.run_server(debug=True)


def generate_colors(int_list):
    """
    Generates a color list based on integer input.
    
    Args:
    int_list (list of int): List of integers.
    
    Returns:
    list of str: List of colors in hex format.
    """
    import matplotlib
    # Normalize the values for scaling
    min_val = min(int_list)
    max_val = max(int_list)
    
    # Create a color map (using a colormap from matplotlib)
    color_map = matplotlib.colormaps['viridis']  # You can choose any other colormap
    
    colors = []
    for value in int_list:
        if value == -1:
            colors.append('grey')  # Assign black for -1
        else:
            # Normalize the value to range [0, 1]
            normalized_value = (value - min_val) / (max_val - min_val) if max_val > min_val else 0
            # Get the color from the colormap and convert to hex
            rgb_color = color_map(normalized_value)
            hex_color = '#' + ''.join(f'{int(c * 255):02x}' for c in rgb_color[:3])  # Exclude alpha channel
            colors.append(hex_color)

    return colors




if __name__ == '__main__':
    main()
