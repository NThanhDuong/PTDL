import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
# from statsmodels.stats.multicomp import pairwise_tukeyhsd
# import statsmodels.api as stats
# from statsmodels.stats.anova import anova_lm
# from statsmodels.formula.api import ols

# Nguồn dữ liệu: https://www.kaggle.com/datasets/nancyalaswad90/diamonds-prices

data = pd.read_csv('DiamondData.csv')
print("==========INFO===================")
print(data.info())
print("==========describe===================")
print(data.describe().to_string())
print("==========nunique===================")
print(data.nunique())  # Check dữ liệu duy nhất, kiểm tra xem thông tin  cơ bản của dữ liệu
print("==========isnull===================")
print(data.isnull().any())  # Kiểm tra xem có dữ liệu null hay không.



# May mắn là dữ liệu này không có dữ liệu null, khá đẹp
# cỘT DỮ LIỆU
# carat: trọng lượng kim cương
# cut: chất lượng chế tác. Tốt nhất là Excellent
# color: màu sắc kim cương. Giá trị nhất là loại D
# clarity: độ tinh khiết. FL là hoàn hảo
# depth: độ dày
# table: độ rộng
# x, y, z: là 3 thông số đo kích thước kim cương. đường kính nhỏ nhất, đường kính lớn nhất, độ sâu đáy
# https://trangsucngoclan.com/tin-tuc/cach-doc-thong-so-tren-ban-chung-nhan-kim-cuong-gia-86


# Kiểm tra dữ liệu các cột kiểu Object
print(data.cut.value_counts())
print(data.color.value_counts())
print(data.clarity.value_counts())

# Số lượng các dòng có giá trị x=0, y=0, z=0
print("=========Số lượng các dòng có giá trị x=0, y=0, z=0==============")
print("So luong: ",len(data.loc[(data.x == 0) | (data.y == 0) | (data.z == 0)].index))

# Xóa giá trị x=0, y=0, z=0
data = data[data['x'] != 0]
data = data[data['y'] != 0]
data = data[data['z'] != 0]
print(data.describe().to_string())

#
# Biểu đồ thể hiện tương quan giữa trọng lượng (carat) và giá (price)
data.plot(kind='scatter', x='carat', y='price')
sns.regplot(data=data, x='carat', y='price', line_kws={'color': '#ff0077'})
#
#
# Biểu đồ thể hiện tương quan giữa độ sâu (depth) và giá (price)
data.plot(kind='scatter', x='depth', y='price')
sns.regplot(data=data, x='depth', y='price', line_kws={'color': '#ff0077'})
#
# Biểu đồ thể hiện tương quan giữa độ rộng (table) và giá (price)
data.plot(kind='scatter', x='table', y='price')
sns.regplot(data=data, x='table', y='price', line_kws={'color': '#ff0077'})

# Biểu đồ thể hiện tương quan giữa đường kính nhỏ nhất (x) và giá (price)
data.plot(kind='scatter', x='x', y='price')
sns.regplot(data=data, x='x', y='price', line_kws={'color': '#ff0077'})

# Biểu đồ thể hiện tương quan giữa đường kính lớn nhất (y) và giá (price)
data.plot(kind='scatter', x='y', y='price')
sns.regplot(data=data, x='y', y='price', line_kws={'color': '#ff0077'})
#
# Biểu đồ thể hiện tương quan giữa độ sâu đáy (z) và giá (price)
data.plot(kind='scatter', x='z', y='price')
sns.regplot(data=data, x='z', y='price', line_kws={'color': '#ff0077'})
#
#biểu đồ Histogram
data[['carat', 'depth', 'x', 'y', 'z', 'table', 'price']].hist(bins=50)

# Đo lệch trái (giá trị âm) / lệch phải (giá trị dương)
print(data[['carat', 'depth', 'table', 'x', 'y', 'z', 'price']].skew())
# Đo phân bố gần trung tâm hay không / đo độ phân tán dữ liệu
print(data[['carat', 'depth', 'table', 'x', 'y', 'z', 'price']].kurt())

# Biểu đồ thể hiện sự liên quan giữa các cột
plt.figure(figsize=(10, 7))
sns.heatmap(data.select_dtypes(include=[np.number]).corr(), annot=True, cmap=plt.cm.Greens)
#
# Biểu đồ Box plot
data[['carat']].plot(kind='box')
data[['depth']].plot(kind='box')
data[['table']].plot(kind='box')
data[['x']].plot(kind='box')
data[['y']].plot(kind='box')
data[['z']].plot(kind='box')
data[['price']].plot(kind='box')

sns.boxplot(x='cut', y='price', data=data)
sns.boxplot(x='color', y='price', data=data)
sns.boxplot(x='clarity', y='price', data=data)

# Phân tích dữ liệu các thuộc tính Cut / Color / Clarity
plt.pie(data.cut.value_counts().values, labels=data.cut.value_counts().index, autopct='%1.1f%%')
plt.title('Cut')
plt.pie(data.color.value_counts().values, labels=data.color.value_counts().index, autopct='%1.1f%%')
plt.title('color')
plt.pie(data.clarity.value_counts().values, labels=data.clarity.value_counts().index, autopct='%1.1f%%')
plt.title('clarity')
#
plt.show()
#
#
# Định danh các loại Cut / Clarity / Color thành các số để dễ dàng xây dựng hàm hồi quy

cut_dict = {'Fair': 1, 'Good': 2, 'Very Good': 3, 'Premium': 4, 'Ideal': 5, }
clarity_dict = {'SI2': 2, 'SI1': 3, 'VS1': 5, 'VS2': 4, 'VVS2': 6, 'VVS1': 7, 'I1': 1, 'IF': 8}
color_dict = {'J': 1, 'I': 2, 'H': 3, 'G': 4, 'F': 5, 'E': 6, 'D': 7, }

data['cut'] = data['cut'].map(cut_dict)
data['clarity'] = data['clarity'].map(clarity_dict)
data['color'] = data['color'].map(color_dict)

# print(data.info())


X = data[['carat', 'x', 'y', 'z', 'depth', 'table', 'cut', 'clarity', 'color']]
Y = data['price']

# XAY DUNG MO HINH HOI QUY
print("==============XAY DUNG MO HINH HOI QUY===============")
lm = linear_model.LinearRegression()
lm.fit(X, Y)
print(lm.score(X, Y))
print(lm.coef_)
print(lm.intercept_)

print(pd.DataFrame(lm.coef_, X.columns))

# mo hinh hoi quy cuoi cung se la
# Y = 3105.51 + 10949.01 * Carat -936.57 * x +  59.80 * y - 104.03 * z -79.26 * depth - 26.76 * table + 120.2 * cut + 499.52 * clarity + 323.22 * color

# Kết quả dự báo
Y_Pred = lm.predict(X)
Y_Pred = pd.DataFrame(Y_Pred, columns=['Price_Predict'])

# So sánh với lương thực tế
print("============So sánh với lương thực tế===========")
result_compare = pd.concat([X, Y, Y_Pred], axis=1)
result_compare['Deviation'] = result_compare['Price_Predict'] - result_compare['price']
print(result_compare)
result_compare.to_csv("output.csv")
print(result_compare['Deviation'].sum())
