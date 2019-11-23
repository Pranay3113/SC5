import pandas as pd
import numpy as np

#%%
df = pd.read_excel("file:///C:/Users/student/Desktop/mba/MBA/Online Retail.xlsx")


#%%
df = df[df.Description.isnull()==False]

df = df[df.InvoiceNo.astype(str).str.contains('C')==False]

df = df[df.Country == 'United Kingdom']


#%%
df2 = df[df.Description.isin(df.Description.value_counts()[:10].index)]
df2.reset_index(inplace=True, drop=True)

df2.drop_duplicates(['InvoiceNo', 'Description'], inplace=True)

#%%
df2_matrix = df2.pivot(index = 'InvoiceNo', columns='Description', values = 'Quantity')


#%%
df2_matrix.fillna(0, inplace=True)

#%%
df2_matrix.values[df2_matrix.values>0] = 1

#%%
#df3_matrix = np.where(np.isnan(df2_matrix.values), 0, 1)
#df4_matrix = pd.DataFrame(df3_matrix, index = df2_matrix.index, columns = df2_matrix.columns)

#%%
def get_support(product_desc: str) -> float:
    return df2_matrix[product_desc].mean()


def get_confidence(product_desc):
    subset = df2_matrix[df2_matrix[product_desc]==1]
    return subset.mean()


def get_lift(product_desc):
    subset = df2_matrix[df2_matrix[product_desc] == 1]
    PAUB = subset.sum()/len(subset)
    PA = df2_matrix[product_desc].sum()/len(df2_matrix[product_desc])
    PB = df2_matrix.sum()/len(df2_matrix)
    return (PAUB)/(PA*PB)


#%%
product_desc = "ASSORTED COLOUR BIRD ORNAMENT"

#%%
support = get_support(product_desc)
conf = get_confidence(product_desc)
lift = get_lift(product_desc)

