import numpy as np
from PIL import Image
import pandas as pd
from scipy.spatial import distance_matrix

img = Image.open('test.png')
img = img.convert("RGB")
from collections import defaultdict
by_color = defaultdict(int)
for pixel in img.getdata():
    by_color[pixel] += 1

colors = []
for pixel in img.getdata():
    if by_color[pixel] > 9 and pixel not in colors:
        colors.append(pixel)

print(len(colors))

"""
col = np.asarray(colors)
np.savetxt("colors.txt", col.astype(int), fmt='%i', delimiter=",")

df = pd.DataFrame(colors)

mat = pd.DataFrame(distance_matrix(df.values, df.values).round(0).astype(int), index=df.index, columns=df.index)

mat.to_csv('out.csv')

np.savetxt("out.txt", mat.values.astype(int), fmt='%i', delimiter=",")

"""
import math
import random

# Google Colab & Kaggle integration
from amplpy import AMPL, ampl_notebook

ampl = AMPL()
# Class for the p-median problem
class PMedianInstance:
    def __init__(
        self,
        num_customers,
        num_facilities,
        p,
    ):
        self.num_customers = num_customers
        self.num_facilities = num_facilities
        self.p = p
        self.facilities = range(num_facilities)
        self.customers = range(num_customers)
        self.customer_coordinates = {}
        self.facility_coordinates = {}
        self.distances = {}
        self.costs = {}
        self.generate_instance(self.d3)

    def generate_instance(self, distance):
        # Generate coordinates for customers and facilities
        for i in self.customers:
            self.customer_coordinates[i] = colors[i]
        for i in self.facilities:
            self.facility_coordinates[i] = colors[i]

        # Calculate distances between each pair of customers and facilities
        for c_id, c_coord in self.customer_coordinates.items():
            for f_id, f_coord in self.facility_coordinates.items():
                self.distances[(c_id, f_id)] = round(distance(c_coord, f_coord))
                self.costs[(c_id, f_id)] = self.distances[(c_id, f_id)]

    def d3(self, coord1, coord2):
        return math.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2 + (coord1[2] - coord2[2]) ** 2)

    def get_p(self):
        return self.p

    def get_customers(self):
        return list(self.customers)

    def get_customers_coordinates(self):
        return self.customer_coordinates

    def get_facilities(self):
        return list(self.facilities)

    def get_facilities_coordinates(self):
        return self.facility_coordinates

    def get_distances(self):
        return self.distances

    def get_distances(self):
        return self.distances

    def get_costs(self):
        return self.costs

    def print_instance(self):
        # Print coordinates of customers and facilities
        print("Customer coordinates:")
        for c_id, c_coord in self.customer_coordinates.items():
            print(f"Customer {c_id}: {c_coord}")

        print("\nFacility coordinates:")
        for f_id, f_coord in self.facility_coordinates.items():
            print(f"Facility {f_id}: {f_coord}")

        # Print costs between each pair of customers and facilities
        print("\nCosts:")
        for (c_id, f_id), cost in self.costs.items():
            print(f"Cost from Customer {c_id} to Facility {f_id}: {cost}")


instance = PMedianInstance(5, 5, 2)
instance.print_instance()

import pandas as pd
import numpy as np


# Prepare data to send to the optimization engine
def prepare_data(num_customers, num_facilities, p):
    instance = PMedianInstance(num_customers, num_facilities, p)
    return (
        instance.get_customers(),
        instance.get_facilities(),
        instance.get_p(),
        instance.get_costs(),
        instance.get_customers_coordinates(),
        instance.get_facilities_coordinates(),
    )


# send data directly from python data structures to ampl
def load_data_2_ampl(model, customers, facilities, p, costs):
    model.set["CUSTOMERS"] = customers
    model.set["FACILITIES"] = facilities
    model.param["p"] = p
    model.param["cost"] = costs


num_customers = len(colors)
num_facilities = len(colors)
p = 10

# read model
ampl.read("pmedian.mod")

# get data
(
    customers,
    facilities,
    p,
    costs,
    customer_coordinates,
    facility_coordinates,
) = prepare_data(num_customers, num_facilities, p)

# load data into ampl
load_data_2_ampl(ampl, customers, facilities, p, costs)

# solve with highs
ampl.solve(solver="highs")
ampl.option["display_eps"] = 1e-2
ampl.display("x, y")

# retrieve dictionaries from ampl with the solution
def retrieve_solution(model):
    # open facilities
    open = model.var["y"].to_dict()
    rounded_open = {key: int(round(value)) for key, value in open.items()}

    costs = model.getData(
        "{i in CUSTOMERS, j in FACILITIES} cost[i,j] * x[i,j]"
    ).to_dict()
    rounded_costs = {
        key: float(round(value, 2))
        for key, value in costs.items()
        if costs[key] >= 5e-6
    }
    return rounded_open, rounded_costs


open_facilities, costs = retrieve_solution(ampl)

print(costs)

d_k = ampl.get_value("total_cost")

print(d_k)

K = 100
ks = 1
d_k1 = 2*d_k - 1

x_o = [[ None for _ in range(num_facilities)] for _ in range(num_customers)]
x_o_v = ampl.get_variable("x")

for i in range(num_customers):
    for j in range(num_facilities):
        x_o[i][j] = x_o_v[i,j].value()

ampl.close()

ampl = ampl_notebook(
    modules=["open"],  # modules to install
    license_uuid="default",  # license to use
)  # instantiate AMPL object

ampl.read("pmedian_2.mod")

# get data
(
    customers,
    facilities,
    p,
    costs,
    customer_coordinates,
    facility_coordinates,
) = prepare_data(num_customers, num_facilities, p)

# load data into ampl
load_data_2_ampl(ampl, customers, facilities, p, costs)

ampl.param["ks"] = ks
ampl.param["K"] = K
ampl.param["d_k1"] = d_k1
xs_v = ampl.param["xs"]

for i in range(num_customers):
    for j in range(num_facilities):
       xs_v[0,i,j] = x_o[i][j]

res = []
res = [x_o]

for k in range(K):
   ampl.solve(solver="highs")
   d_k = ampl.get_value("total_cost")
   xs_v = ampl.param["xs"]
   x_v = ampl.get_variable("x")
   if (d_k > round((d_k1+1)/2)): break

   xs = [[[ None for _ in range(num_facilities)] for _ in range(num_customers)] for _ in range( K )]
   x = [[ None for _ in range(num_facilities)] for _ in range(num_customers)]

   for c in range(ks):
    for i in range(num_customers):
        for j in range(num_facilities):
            xs[c][i][j] = xs_v[c,i,j]

   for i in range(num_customers):
    for j in range(num_facilities):
        x[i][j] = x_v[i,j].value()

   dist = 0
   for i in range(num_customers):
      for j in range(num_facilities):
        dist = dist + abs(xs[k][i][j] - x[i][j])

   if (dist < 1): break
   
   for i in range(num_customers):
    for j in range(num_facilities):
        x[i][j] = round(x[i][j])

   xs[k+1] = x

   ks = ks + 1
   
   for i in range(num_customers):
    for j in range(num_facilities):
       xs_v[k+1,i,j] = xs[k+1][i][j]

   ampl.param["ks"] = ks

   res = xs

for c in range(ks):
    print("\n", c)
    for i in range(num_customers):
        print(res[c][i])

#ampl.display("xs")