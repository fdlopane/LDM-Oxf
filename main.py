"""
Land Development Model (LDM) code developed by Michael Batty and Fulvio D. Lopane
Centre for Advanced Spatial Analysis
University College London

Main module - generates maps and main outputs

Started developing in May 2024
"""

########################################################################################################################
# Import phase

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import *

########################################################################################################################
# Read layers input files
df1  = pd.read_csv(inputs["conservation_area_csv"])
df2  = pd.read_csv(inputs["flood_risk_area_csv"])
df3  = pd.read_csv(inputs["greenbelt_csv"])
df4  = pd.read_csv(inputs["highway_coverage_csv"])
df5  = pd.read_csv(inputs["historical_conservation_csv"])
df6  = pd.read_csv(inputs["HA_bus_csv"])
df7  = pd.read_csv(inputs["HA_rail_csv"])
df8  = pd.read_csv(inputs["HA_road_csv"])
df9  = pd.read_csv(inputs["JA_bus_csv"])
df10 = pd.read_csv(inputs["JA_rail_csv"])
df11 = pd.read_csv(inputs["JA_road_csv"])
df12 = pd.read_csv(inputs["national_nature_reserves_csv"])
df13 = pd.read_csv(inputs["ONB_csv"])
df14 = pd.read_csv(inputs["parks_and_gardens_csv"])
df15 = pd.read_csv(inputs["population_density_csv"])
df16 = pd.read_csv(inputs["proximity_to_green_space_csv"])
df17 = pd.read_csv(inputs["special_scientific_interest_csv"])
df18 = pd.read_csv(inputs["slope_csv"])
df19 = pd.read_csv(inputs["surface_water_csv"])
dfc  = pd.read_csv(inputs["coordinates_csv"])

# Number of pixels
N = 353

xcoord = dfc[['x']].to_numpy()
ycoord = dfc[['y']].to_numpy()

# Convert the input layers:
value1 = df1[['value']].to_numpy()
value2 = df2[['value']].to_numpy()
value3 = df3[['value']].to_numpy()
value4 = df4[['value']].to_numpy()
value5 = df5[['value']].to_numpy()
value6 = df6[['value']].to_numpy()
value7 = df7[['value']].to_numpy()
value8 = df8[['value']].to_numpy()
value9 = df9[['value']].to_numpy()
value10 = df10[['value']].to_numpy()
value11 = df11[['value']].to_numpy()
value12 = df12[['value']].to_numpy()
value13 = df13[['value']].to_numpy()
value14 = df14[['value']].to_numpy()
value15 = df15[['value']].to_numpy()
value16 = df16[['value']].to_numpy()
value17 = df17[['value']].to_numpy()
value18 = df18[['value']].to_numpy()
value19 = df19[['value']].to_numpy()

X = np.full((N),1.0)
Y = np.full((N),1.0)

for i in range(0,N):
    X[i] = xcoord[i]
    Y[i] = ycoord[i]

########################################################################################################################
# Normalise the values of each layer

for i in range(0, N):
    value1[i] = 100 * ((value1[i]-min(value1))/(max(value1)-min(value1)))
    value2[i] = 100 * ((value2[i]-min(value2))/(max(value2)-min(value2)))
    value3[i] = 100 * ((value3[i]-min(value3))/(max(value3)-min(value3)))
    value4[i] = 100 * ((value4[i]-min(value4))/(max(value4)-min(value4)))
    value5[i] = 100 * ((value5[i]-min(value5))/(max(value5)-min(value5)))
    value6[i] = 100 * ((value6[i]-min(value6))/(max(value6)-min(value6)))
    value7[i] = 100 * ((value7[i]-min(value7))/(max(value7)-min(value7)))
    value8[i] = 100 * ((value8[i]-min(value8))/(max(value8)-min(value8)))
    value9[i] = 100 * ((value9[i]-min(value9))/(max(value9)-min(value9)))
    value10[i] = 100 * ((value10[i]-min(value10))/(max(value10)-min(value10)))
    value11[i] = 100 * ((value11[i]-min(value11))/(max(value11)-min(value11)))
    value12[i] = 100 * ((value12[i]-min(value12))/(max(value12)-min(value12)))
    value13[i] = 100 * ((value13[i]-min(value13))/(max(value13)-min(value13)))
    value14[i] = 100 * ((value14[i]-min(value14))/(max(value14)-min(value14)))
    value15[i] = 100 * ((value15[i]-min(value15))/(max(value15)-min(value15)))
    value16[i] = 100 * ((value16[i]-min(value16))/(max(value16)-min(value16)))
    value17[i] = 100 * ((value17[i]-min(value17))/(max(value17)-min(value17)))
    value18[i] = 100 * ((value18[i]-min(value18))/(max(value18)-min(value18)))
    value19[i] = 100 * ((value19[i]-min(value19))/(max(value19)-min(value19)))

v1 = np.full((N), 1.0)
v2 = np.full((N), 1.0)
v3 = np.full((N), 1.0)
v4 = np.full((N), 1.0)
v5 = np.full((N), 1.0)
v6 = np.full((N), 1.0)
v7 = np.full((N), 1.0)
v8 = np.full((N), 1.0)
v9 = np.full((N), 1.0)
v10 = np.full((N), 1.0)
v11 = np.full((N), 1.0)
v12 = np.full((N), 1.0)
v13 = np.full((N), 1.0)
v14 = np.full((N), 1.0)
v15 = np.full((N), 1.0)
v16 = np.full((N), 1.0)
v17 = np.full((N), 1.0)
v18 = np.full((N), 1.0)
v19 = np.full((N), 1.0)

for i in range(0 ,N):
    v1[i] = 100 * ((value1[i] - min(value1)) / (max(value1) - min(value1)))
    v2[i] = 100 * ((value2[i] - min(value2)) / (max(value2) - min(value2)))
    v3[i] = 100 * ((value3[i] - min(value3)) / (max(value3) - min(value3)))
    v4[i] = 100 * ((value4[i] - min(value4)) / (max(value4) - min(value4)))
    v5[i] = 100 * ((value5[i] - min(value5)) / (max(value5) - min(value5)))
    v6[i] = 100 * ((value6[i] - min(value6)) / (max(value6) - min(value6)))
    v7[i] = 100 * ((value7[i] - min(value7)) / (max(value7) - min(value7)))
    v8[i] = 100 * ((value8[i] - min(value8)) / (max(value8) - min(value8)))
    v9[i] = 100 * ((value9[i] - min(value9)) / (max(value9) - min(value9)))
    v10[i] = 100 * ((value10[i] - min(value10)) / (max(value10) - min(value10)))
    v11[i] = 100 * ((value11[i] - min(value11)) / (max(value11) - min(value11)))
    v12[i] = 100 * ((value12[i] - min(value12)) / (max(value12) - min(value12)))
    v13[i] = 100 * ((value13[i] - min(value13)) / (max(value13) - min(value13)))
    v14[i] = 100 * ((value14[i] - min(value14)) / (max(value14) - min(value14)))
    v15[i] = 100 * ((value15[i] - min(value15)) / (max(value15) - min(value15)))
    v16[i] = 100 * ((value16[i] - min(value16)) / (max(value16) - min(value16)))
    v17[i] = 100 * ((value17[i] - min(value17)) / (max(value17) - min(value17)))
    v18[i] = 100 * ((value18[i] - min(value18)) / (max(value18) - min(value18)))
    v19[i] = 100 * ((value19[i] - min(value19)) / (max(value19) - min(value19)))

########################################################################################################################
# Plot maps

series = np.full((N), 1.0)

q = 0.6

for j in range(1, 20):
    plt.axes().set_aspect('equal')
    plt.xticks(visible=False)
    plt.yticks(visible=False)
    plt.tick_params(left = False, bottom = False)
    if j==1: series = q*v1+1
    if j==2: series = q*v2+1
    if j==3: series = q*v3+1
    if j==4: series = q*v4+1
    if j==5: series = q*v5+1
    if j==6: series = q*v6+1
    if j==7: series = q*v7+1
    if j==8: series = q*v8+1
    if j==9: series = q*v9+1
    if j==10: series = q*v10+1
    if j==11: series = q*v11+1
    if j==12: series = q*v12+1
    if j==13: series = q*v13+1
    if j==14: series = q*v14+1
    if j==15: series = q*v15+1
    if j==16: series = q*v16+1
    if j==17: series = q*v17+1
    if j==18: series = q*v18+1
    if j==19: series = q*v19+1

    plt.scatter(X, Y, s=series, alpha=1.0, color='black')

    if j==1: plt.savefig(outputs["Conservation_Areas"])
    if j==2: plt.savefig(outputs["Flood_Risk_Areas"])
    if j==3: plt.savefig(outputs["Greenbelt"])
    if j==4: plt.savefig(outputs["Highway_Coverage"])
    if j==5: plt.savefig(outputs["Historical_Conservation"])
    if j==6: plt.savefig(outputs["Housing_Accessibility_by_Bus"])
    if j==7: plt.savefig(outputs["Housing_Accessibility_by_Rail"])
    if j==8: plt.savefig(outputs["Housing_Accessibility_by_Road"])
    if j==9: plt.savefig(outputs["Job_Accessibility_by_Bus"])
    if j==10: plt.savefig(outputs["Job_Accessibility_by_Rail"])
    if j==11: plt.savefig(outputs["Job_Accessibility_by_Road"])
    if j==12: plt.savefig(outputs["National_Nature_Reserves"])
    if j==13: plt.savefig(outputs["Areas_of_Outstanding_Natural_Beauty"])
    if j==14: plt.savefig(outputs["Parks_and_Gardens"])
    if j==15: plt.savefig(outputs["Population_Density"])
    if j==16: plt.savefig(outputs["Proximity_to_Green_Space"])
    if j==17: plt.savefig(outputs["Special_Scientific_Interest"])
    if j==18: plt.savefig(outputs["Slope"])
    if j==19: plt.savefig(outputs["Surface_Water"])

# Equal weights (average)
Av = np.full((N), 1.0)
Av = v1+v2+v3+v4+v5+v6+v7+v8+v9+v10+v11+v12+v13+v14+v15+v16+v17+v18+v19
weight = np.full((19), 1.0)

for i in range(0, 19):
    weight[i] = 1/19
weight.tofile(outputs["Weights_equal"], sep=',')

for i in range(0, N):
    Av[i] = Av[i]/N

for i in range(0, N):
    Av[i] = 100 * ((Av[i]-min(Av))/(max(Av)-min(Av)))

plt.axes().set_aspect('equal')
plt.xticks(visible=False)
plt.yticks(visible=False)
plt.tick_params(left=False, bottom=False)
series = q*Av+1

series.tofile(outputs["Av_weights_map_values"], sep=',')

plt.scatter(X, Y, s=series, alpha=1.0, color='black')
plt.savefig(outputs["Av_weights_map"])

# Hierarchical weights
j=100

w1 = 1/16
w2 = 1/8
w3 = 1/64
w4 = 1/128
w5 = 1/16
w6 = 1/8
w7 = 1/32
w8 = 1/64
w9 = 1/64
w10 = 1/128
w11 = 1/32
w12 = 1/16
w13 = 1/16
w14 = 1/8
w15 = 1/64
w16 = 1/32
w17 = 1/64
w18 = 1/16
w19 = 1/8

weight[0] = w1
weight[1] = w2
weight[2] = w3
weight[3] = w4
weight[4] = w5
weight[5] = w6
weight[6] = w7
weight[7] = w8
weight[8] = w9
weight[9] = w10
weight[10] = w11
weight[11] = w12
weight[12] = w13
weight[13] = w14
weight[14] = w15
weight[15] = w16
weight[16] = w17
weight[17] = w18
weight[18] = w19

totalw=w1+w2+w3+w4+w5+w6+w7+w8+w9+w10+w11+w12+w13+w14+w15+w16+w17+w18+w19

weight.tofile(outputs["Weights_hierarchical"], sep=',')

AvD = np.full((N),1.0)
AvDD = np.full((N),1.0)
AvD = (w1*v1)+(w2*v2)+(w3*v3)+(w4*v4)+(w5*v5)+(w6*v6)+(w7*v7)+(w8*v8)+(w9*v9)+(w10*v10)+(w11*v11)+(w12*v12)+(w13*v13)+(w14*v14)+(w15*v15)+(w16*v16)+(w17*v17)+(w18*v18)+(w19*v19)

for i in range(0,N):
    AvD[i] = AvD[i]/N

for i in range(0,N):
    AvD[i] = 100 * ((AvD[i]-min(AvD))/(max(AvD)-min(AvD)))


plt.axes().set_aspect('equal')
plt.xticks(visible=False)
plt.yticks(visible=False)
plt.tick_params(left=False, bottom=False)
series = q*AvD+1

series.tofile(outputs["Hierarchical_weights_map_values"], sep=',')

plt.scatter(X, Y, s=series, alpha=1.0, color = 'black')
plt.savefig(outputs["Hierarchical_weights_map"])

for i in range(0, N):
    AvD[i] = 100 * ((AvD[i]-min(AvD))/(max(AvD)-min(AvD)))
AvDD = AvD
beta = 3
for i in range(0, N):
    AvDD[i] = (AvDD[i])**beta

for i in range(0,N):
    AvDD[i] = 100 * ((AvDD[i]-min(AvDD))/(max(AvDD)-min(AvDD)))

plt.axes().set_aspect('equal')
plt.xticks(visible=False)
plt.yticks(visible=False)
plt.tick_params(left=False, bottom=False)
series = q*AvDD+1

plt.scatter(X, Y, s=series, alpha=1.0, color='black')
plt.savefig(outputs["Hierarchical_weights_cube_map"])

# Weights from correlation matrix

dfname = pd.read_csv(inputs["correlation_matrix_csv"])

correlations = dfname.corr()

cor = correlations.values

a = np.full((cor.shape[0]), 1.0)
mm = 200

for i in range(cor.shape[0]):
    for j in range(cor.shape[1]):
        if cor[i][j] <= mm:
            mm = cor[i][j]

for i in range(0, cor.shape[1]):
    for j in range(0, cor.shape[1]):
        cor[i][j] = cor[i][j] - mm

for i in range(cor.shape[0]):
    sum = 0.0
    for j in range(cor.shape[1]):
        sum = sum + cor[i][j]
    a[i] = sum

    for j in range(cor.shape[1]):
        cor[i][j] = cor[i][j] / a[i]

for i in range(0, cor.shape[1]):
    sum = 0.0
    for j in range(0, cor.shape[1]):
        sum = sum + cor[i][j]

c1 = cor

for k in range(0, 20):
    c1 = np.matmul(c1, cor)

w1 = c1[1][0]
w2 = c1[1][1]
w3 = c1[1][2]
w4 = c1[1][3]
w5 = c1[1][4]
w6 = c1[1][5]
w7 = c1[1][6]
w8 = c1[1][7]
w9 = c1[1][8]
w10 = c1[1][9]
w11 = c1[1][10]
w12 = c1[1][11]
w13 = c1[1][12]
w14 = c1[1][13]
w15 = c1[1][14]
w16 = c1[1][15]
w17 = c1[1][16]
w18 = c1[1][17]
w19 = c1[1][18]

weight[0] = w1
weight[1] = w2
weight[2] = w3
weight[3] = w4
weight[4] = w5
weight[5] = w6
weight[6] = w7
weight[7] = w8
weight[8] = w9
weight[9] = w10
weight[10] = w11
weight[11] = w12
weight[12] = w13
weight[13] = w14
weight[14] = w15
weight[15] = w16
weight[16] = w17
weight[17] = w18
weight[18] = w19

weight.tofile(outputs["Weights_cor"], sep=',')

DesignAvD = np.full((N), 1.0)
DesignAvD = (w1*v1)+(w2*v2)+(w3*v3)+(w4*v4)+(w5*v5)+(w6*v6)+(w7*v7)+(w8*v8)+(w9*v9)+(w10*v10)+(w11*v11)+(w12*v12)+(w13*v13)+(w14*v14)+(w15*v15)+(w16*v16)+(w17*v17)+(w18*v18)+(w19*v19)

for i in range(0, N):
    DesignAvD[i] = DesignAvD[i] / N

#DesignAvD[i]

beta = 3
for i in range(0, N):
    DesignAvD[i]=(DesignAvD[i])**beta

for i in range(0, N):
    DesignAvD[i] = 100 * ((DesignAvD[i]-min(DesignAvD))/(max(DesignAvD)-min(DesignAvD)))

plt.axes().set_aspect('equal')
plt.xticks(visible=False)
plt.yticks(visible=False)
plt.tick_params(left=False, bottom=False)
series = q*DesignAvD+1

series.tofile(outputs["Weights_cor_map_values"], sep=',')

plt.scatter(X, Y, s=series, alpha=1.0, color = 'black')
plt.savefig(outputs["Weights_cor_map"])

for i in range(0, N):
    DesignAvD[i] = 100 * ((DesignAvD[i]-min(DesignAvD))/(max(DesignAvD)-min(DesignAvD)))

plt.axes().set_aspect('equal')
plt.xticks(visible=False)
plt.yticks(visible=False)
plt.tick_params(left=False, bottom=False)
series = q*DesignAvD+1

series.tofile(outputs["Weights_cor_cube_map_values"], sep=',')

plt.scatter(X, Y, s=series, alpha=1.0, color = 'black')
plt.savefig(outputs["Weights_cor_cube_map"])
