##
import pandas as pd
import numpy as np

##
def append_datasets_row(csv1,csv2): #two csv files for proteomics. Ratio between 0 to 10.
    # processing protein spreadsheets.
    protFrame1 = pd.read_csv(csv1)
    protFrame2 = pd.read_csv(csv2)
    protFrame2 = protFrame2.iloc[10:, :]
    train_protFrame = pd.concat([protFrame1, protFrame2],ignore_index=True)
    for i in range(len(train_protFrame)):
        if train_protFrame.iloc[i,5] == "Long covid":
            train_protFrame.iloc[i,5] = 1;
        else:
            train_protFrame.iloc[i,5] = 0;


    new_train_protFrame = train_protFrame.reset_index(drop=True)
    new_train_protFrame = new_train_protFrame.iloc[:,1:]
    new_train_protFrame.to_csv('combined_LC_NLC_Prot.csv', index=False)
    print(new_train_protFrame.iloc[71,0])
    return new_train_protFrame


testdata = append_datasets_row("data/4-4-22_Test.AllLC_vsHealthy_Data.csv","data/4-4-22_Test.AllNLC_vsHealthy_Data.csv")

##

# step 1: make table with only 2 columns and turn into dictionary (map id1 to id2)
df = pd.read_csv("data/step1meta.csv")
df1 = df.iloc[:,0]
df2 = df.iloc[:,8]
df3 = pd.concat([df1,df2],axis=1)
dictionary1 = dict(df3.values)
print(df3)
print(dictionary1)

# step 2: map id1 to id3 by replacing id2
df_2 = pd.read_csv("data/step2meta.csv")

for key in dictionary1:
    dfnew = df_2.loc[df_2['Sample ID for shipment/Location in shipment box'] == dictionary1[key]]
    dictionary1[key] = dfnew.iloc[0,0]

print(dictionary1)

# step 3: gather row index values for key/value pair
df_3 = pd.read_csv("data/step3meta.csv")

clev = np.zeros(72)
pID = np.zeros(72)
counter = 0

for key in dictionary1:
    ind1 = testdata.index[testdata['Participant.ID'] == dictionary1[key]].tolist()
    ind1 = ind1[0]

    ind2 = df_3.index[df_3['PARENT_SAMPLE_NAME'] == key].tolist()
    ind2 = ind2[0]

    pID[counter] = ind1
    clev[counter] = ind2
    counter = counter + 1

# step 4: order id1 from least to greatest, moving the index values for id3 alongside
sort = np.argsort(pID)
clev = clev[sort]

print(clev)
# step 5: select based on the sorted indices and concatenate axis 1 (columns) the metabolomic and proteomic datasets

df_meta = df_3.iloc[clev].reset_index(drop=True)

# getting rid of columns without unique values
df_meta_2 = df_meta.drop(df_meta.std()[(df_meta.std() == 0)].index, axis=1).reset_index(drop=True)
print(np.shape(df_meta))
print(np.shape(df_meta_2))

final_df = pd.concat([testdata.reset_index(drop=True), df_meta_2.iloc[:,1:-3]],axis=1)
newfinaldf = final_df.reset_index(drop=True)
newfinaldf2 = newfinaldf.loc[newfinaldf['Timepoint'] != '3']
newfinaldf3 = newfinaldf2.loc[newfinaldf2['Timepoint'] != '4']
newfinaldf4 = newfinaldf3.loc[newfinaldf3['Timepoint'] != '5']

newfinaldf5 = newfinaldf4.reset_index(drop=True)
print(np.where(newfinaldf5.std() == 0)[0])
df6 = newfinaldf5.drop(newfinaldf5.std()[(newfinaldf5.std() == 0)].index,axis = 1).reset_index(drop=True)

print(np.shape(df6))
print(pd.concat([df6.iloc[:,4],df6.iloc[:,6:]],axis=1))
df6.to_csv('Prot_Meta_Combined2.csv',index=False)

## making early data shape (missing one column) with all timepoints.