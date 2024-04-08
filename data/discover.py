import dowhy
from dowhy import CausalModel
import numpy as np
import pandas as pd
import networkx as nx
import os
import graphviz
from causallearn.search.FCMBased.lingam.utils import make_dot
from causallearn.search.FCMBased import lingam
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ScoreBased.GES import ges
from causallearn.search.HiddenCausal.GIN.GIN import GIN
from causallearn.utils.GraphUtils import GraphUtils
import dowhy.datasets
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import io

np.set_printoptions(precision=3, suppress=True)
np.random.seed(0)


# Custom Causal Graph
causal_graph_y = """
digraph {
dis -> y;
age -> dis; age -> y;
CT_R -> dis; CT_R -> y;
CT_E -> dis; CT_E -> y;
danger -> dis; danger -> y;
gender -> dis; gender -> y;
is_korean -> dis; is_korean -> y;
primary case -> dis; primary case -> y;
job_idx -> dis; job_idx -> y;
rep_idx -> dis; rep_idx -> y;
place_idx -> dis; place_idx -> y;
add_idx -> dis; add_idx -> y;
}
"""

causal_graph_d = """
digraph {
dis -> d;
age -> dis; age -> d;
CT_R -> dis; CT_R -> d;
CT_E -> dis; CT_E -> d;
danger -> dis; danger -> d;
gender -> dis; gender -> d;
is_korean -> dis; is_korean -> d;
primary case -> dis; primary case -> d;
job_idx -> dis; job_idx -> d;
rep_idx -> dis; rep_idx -> d;
place_idx -> dis; place_idx -> d;
add_idx -> dis; add_idx -> d;
}
"""

def make_graph(adjacency_matrix, labels=None):
    idx = np.abs(adjacency_matrix) > 0.01
    dirs = np.where(idx)
    d = graphviz.Digraph(engine='dot')
    names = labels if labels else [f'x{i}' for i in range(len(adjacency_matrix))]
    for name in names:
        d.node(name)
    for to, from_, coef in zip(dirs[0], dirs[1], adjacency_matrix[idx]):
        d.edge(names[from_], names[to], label=str(coef))
    return d

def str_to_dot(string):
    '''
    Converts input string from graphviz library to valid DOT graph format.
    '''
    graph = string.strip().replace('\n', ';').replace('\t','')
    graph = graph[:9] + graph[10:-2] + graph[-1] # Removing unnecessary characters from string
    return graph

# Let's generate some "normal" data we assume we're given from our problem domain:
for cut_date in range(6):
    for tag in ['', '_simple']:
        data_path = f'./data_cut_{cut_date}.csv'
        data = pd.read_csv(data_path)
        data = data.drop(['cut_date','diff_days'], axis=1)
        if tag == '_simple':
            data = data[['dis','danger', 'y', 'd']]
        labels = [f'{col}' for i, col in enumerate(data.columns)]
        tag=f'{tag}_{cut_date}'
        for causal_graph in [causal_graph_y, causal_graph_d]:
            dot_source = graphviz.Source(causal_graph)
            if causal_graph == causal_graph_y:
                if not os.path.exists(f'./fig/custom_graph_y.png'):
                    dot_source.render(f'./fig/custom_graph_y', format='png', engine='dot')
            else:
                if not os.path.exists(f'./fig/custom_graph_d.png'):
                    dot_source.render(f'./fig/custom_graph_d', format='png', engine='dot')
            
            # DoWhy 모델 설정
            model = CausalModel(
                data = data,
                treatment = 'dis',
                outcome = 'y' if causal_graph == causal_graph_y else 'd',
                common_causes = data.columns.difference(['dis', 'y']).tolist() if causal_graph == causal_graph_y else data.columns.difference(['dis', 'd']).tolist() ,
                graph = causal_graph.replace("\n", " ")
            )

            # 인과 추론을 위한 모델 식별
            identified_estimand = model.identify_effect()

            for method in ['backdoor.propensity_score_matching',
                        'backdoor.propensity_score_stratification',
                        'backdoor.propensity_score_weighting',
                        'backdoor.linear_regression',
                        'iv.regression_discontinuity',
                        'iv.instrumental_variable',
                        'frontdoor.two_stage_regression'
                        ] :
                estimate = model.estimate_effect(identified_estimand,
                                                method_name=method)

                print(f'method : {method} \n {estimate}')
            
        if not os.path.exists(f'./fig/pc{tag}.png'):
            print("discovering with pc algorithm")
            data = data.to_numpy()

            cg = pc(data)

            # Visualization using pydot
            pyd = GraphUtils.to_pydot(cg.G, labels=labels)
            tmp_png = pyd.create_png(f="png")
            fp = io.BytesIO(tmp_png)
            img = mpimg.imread(fp, format='png')
            plt.axis('off')
            plt.imshow(img)
            plt.savefig(f'./fig/pc{tag}.png', dpi=300)


        if not os.path.exists(f'./fig/ges{tag}.png'):
            print("discovering with ges algorithm")
            # default parameters
            Record = ges(data)

            # Visualization using pydot
            pyd = GraphUtils.to_pydot(Record['G'], labels=labels)
            tmp_png = pyd.create_png(f="png")
            fp = io.BytesIO(tmp_png)
            img = mpimg.imread(fp, format='png')
            plt.axis('off')
            plt.imshow(img)
            plt.savefig(f'./fig/ges{tag}.png', dpi=300)


        if not os.path.exists(f'./fig/lingam{tag}.png'):
            print("discovering with lingam algorithm")
            model = lingam.ICALiNGAM()
            model.fit(data)

            
            dot_object = make_dot(model.adjacency_matrix_, labels=labels)


            # 'dot_object'는 'Digraph' 객체입니다.
            dot_str = dot_object.source  # DOT 언어로 변환

            # DOT 문자열을 사용하여 Source 객체 생성
            dot_source = graphviz.Source(dot_str)

            # 이미지 파일로 렌더링하여 저장
            dot_source.render(f'./fig/lingam{tag}', format='png', engine='dot')



# for cut_date in range(6):
#     for tag in ['', '_simple']:
#         data_path = f'./data_cut_{cut_date}.csv'
#         data = pd.read_csv(data_path)
#         data = data.drop(['cut_date','diff_days'], axis=1)
#         labels = [f'{col}' for i, col in enumerate(data.columns)]
#         if tag == '_simple':
#             data = data[['dis', 'y', 'd']]
#         tag=f'{tag}_{cut_date}'
#         # Visualization using pydot
#         if not os.path.exists(f'./fig/gin{tag}.png'):
#             print("discovering with gin algorithm")
#             if isinstance(data, pd.DataFrame):
#                 data = data.to_numpy()
#             G, K = GIN(data)
#             pyd = GraphUtils.to_pydot(G)
#             tmp_png = pyd.create_png(f="png")
#             fp = io.BytesIO(tmp_png)
#             img = mpimg.imread(fp, format='png')
#             plt.axis('off')
#             plt.imshow(img)
#             plt.savefig(f'./fig/gin{tag}.png', dpi=300)