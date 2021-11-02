
# coding: utf-8

# In[3]:


import pydotplus as pdp
from IPython.display import display, Image
graph = pdp.graph_from_dot_data('digraph g{ a -> b -> c; c ->a; a[shape=box,label="A",fillcolor="pink",style=filled]; b[style=filled, fillcolor="yellow"]; c[style=filled, fillcolor="green"];}')
graph1 = pdp.graph_from_dot_data('graph demo1{ a--b; b--c; c--a; }')
display(Image(graph.create_png()))
display(Image(graph1.create_png()))


# In[4]:


get_ipython().system(' pip install graphviz')


# In[8]:


from graphviz import Digraph
dot = Digraph(comment='testing')
dot.node('A')
dot

