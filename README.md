# Below is the Python code that was used in the data mining project and Machine learning of Abalone dataset:

# Importing needed libraries for Data Mining Mini Project in Python 

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, classification_report

# Data Pre-processing
## Data Importing

```python
# Loading the given datasets
data_2020 = pd.read_csv(r'C:\Users\User\OneDrive\Documents\Abalone Data 20200.csv')
data_2021 = pd.read_csv(r'C:\Users\User\OneDrive\Documents\Abalone Data 2021.csv')

data = pd.read_csv(r'C:\Users\User\OneDrive\Documents\Abalone Data 2021.csv')

# To the datasets if it is well imported 

data_2020.head()
data_2021.head()

```


```python
data_2020.dtypes
data_2021.dtypes

```

    Sex                  object
    Length(mm)          float64
    Diameter(mm)        float64
    Height(mm)          float64
    WholeWeight(g)      float64
    ShuckedWeight(g)    float64
    SellWeight(g)       float64
    Spots               float64
    dtype: object



#Check for null values in the datasets
data_2020.isnull().sum() # Null value in 2020 dataset


```python
data_2020.isnull().sum() # Null value in 2020 dataset

```




    Sex                 0
    Length(mm)          2
    Diameter(mm)        1
    Height(mm)          1
    WholeWeight(g)      1
    ShuckedWeight(g)    2
    SellWeight(g)       1
    Spots               0
    dtype: int64




```python
data_2021.isnull().sum() # Null value in 2020 dataset
```




    Sex                 0
    Length(mm)          3
    Diameter(mm)        2
    Height(mm)          2
    WholeWeight(g)      4
    ShuckedWeight(g)    1
    SellWeight(g)       2
    Spots               1
    dtype: int64



Missing Value Handling: Checking and treating of missing values 
The missing value was check and counted and filled with median imputation so as not to influence the outliers.



```python

# Treating the missing values 

from sklearn.impute import SimpleImputer

# Using the median to fill the missiing so as not to influence the outlier
Numeric_Columns = ['Length(mm)', 'Diameter(mm)', 'Height(mm)', 'WholeWeight(g)', 'ShuckedWeight(g)', 'SellWeight(g)', 'Spots']

# C imputer object 
imputer = SimpleImputer(strategy='median')

# Fitting and transforming the imputer for the numerical columns in both datasets
data_2020[Numeric_Columns] = imputer.fit_transform(data_2020[Numeric_Columns])
data_2021[Numeric_Columns] = imputer.fit_transform(data_2021[Numeric_Columns])



```


```python
# Handling Sex Column using Label Encoding for 'Sex' (M, F, I)
le = LabelEncoder()
data_2020['Sex'] = le.fit_transform(data_2020['Sex'])
data_2021['Sex'] = le.transform(data_2021['Sex'])


```


```python
# Add 'Year' column
data_2020['Year'] = 2020
data_2021['Year'] = 2021
```


```python
# Combining the datasets
Combined_data = pd.concat([data_2020, data_2021], ignore_index=True)

```


```python
Combined_data.head
```




    <bound method NDFrame.head of      Sex  Length(mm)  Diameter(mm)  Height(mm)  WholeWeight(g)  \
    0      M       0.455         0.365       0.095          0.5140   
    1      M       0.350         0.265       0.090          0.2255   
    2      F       0.530         0.420       0.135          0.6770   
    3      M       0.440         0.365       0.125          0.5160   
    4      I       0.330         0.255       0.080          0.2050   
    ...   ..         ...           ...         ...             ...   
    4172   F       0.565         0.450       0.165          0.8870   
    4173   M       0.590         0.440       0.135          0.9660   
    4174   M       0.600         0.475       0.000          1.1760   
    4175   F       0.625         0.485       0.150          1.0945   
    4176   M       0.710         0.555       0.195          1.9485   
    
          ShuckedWeight(g)  SellWeight(g)  Spots  Year  
    0               0.2245         0.1500   15.0  2020  
    1               0.0995         0.0700    7.0  2020  
    2               0.2565         0.2100    9.0  2020  
    3               0.2155         0.1550   10.0  2020  
    4               0.0895         0.0550    7.0  2020  
    ...                ...            ...    ...   ...  
    4172            0.3700         0.2490   11.0  2021  
    4173            0.4390         0.2605   10.0  2021  
    4174            0.5255         0.3080    9.0  2021  
    4175            0.5310         0.2960   10.0  2021  
    4176            0.9455         0.0000   12.0  2021  
    
    [4177 rows x 9 columns]>



2.3	Descriptive Statistics/ Exploratory Data Analysis (EDA):
 The summary statistics for each feature (mean, median, standard deviation) of the combined datasets was computed in order Year and gender for clear comparison and exported to csv file.



```python
# descriptive statistics by year and sex
Descriptive_Combined_data = Combined_data.groupby(['Year', 'Sex']).describe()

Descriptive_Combined_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th colspan="8" halign="left">Length(mm)</th>
      <th colspan="2" halign="left">Diameter(mm)</th>
      <th>...</th>
      <th colspan="2" halign="left">SellWeight(g)</th>
      <th colspan="8" halign="left">Spots</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
      <th>count</th>
      <th>mean</th>
      <th>...</th>
      <th>75%</th>
      <th>max</th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>Year</th>
      <th>Sex</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3" valign="top">2020</th>
      <th>F</th>
      <td>641.0</td>
      <td>0.577590</td>
      <td>0.089678</td>
      <td>0.000</td>
      <td>0.5250</td>
      <td>0.585</td>
      <td>0.640</td>
      <td>0.815</td>
      <td>641.0</td>
      <td>0.454610</td>
      <td>...</td>
      <td>0.380000</td>
      <td>1.005</td>
      <td>641.0</td>
      <td>11.152886</td>
      <td>3.289428</td>
      <td>0.0</td>
      <td>9.0</td>
      <td>10.0</td>
      <td>12.0</td>
      <td>29.0</td>
    </tr>
    <tr>
      <th>I</th>
      <td>683.0</td>
      <td>0.425132</td>
      <td>0.110049</td>
      <td>0.075</td>
      <td>0.3575</td>
      <td>0.435</td>
      <td>0.515</td>
      <td>0.680</td>
      <td>683.0</td>
      <td>0.323975</td>
      <td>...</td>
      <td>0.174000</td>
      <td>0.530</td>
      <td>683.0</td>
      <td>7.805271</td>
      <td>2.514795</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>7.0</td>
      <td>9.0</td>
      <td>21.0</td>
    </tr>
    <tr>
      <th>M</th>
      <td>773.0</td>
      <td>0.558286</td>
      <td>0.104654</td>
      <td>0.155</td>
      <td>0.5050</td>
      <td>0.575</td>
      <td>0.630</td>
      <td>0.775</td>
      <td>773.0</td>
      <td>0.436902</td>
      <td>...</td>
      <td>0.353500</td>
      <td>0.897</td>
      <td>773.0</td>
      <td>10.632600</td>
      <td>3.041096</td>
      <td>3.0</td>
      <td>9.0</td>
      <td>10.0</td>
      <td>12.0</td>
      <td>26.0</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">2021</th>
      <th>F</th>
      <td>666.0</td>
      <td>0.579782</td>
      <td>0.085351</td>
      <td>0.290</td>
      <td>0.5250</td>
      <td>0.595</td>
      <td>0.640</td>
      <td>0.800</td>
      <td>666.0</td>
      <td>0.453041</td>
      <td>...</td>
      <td>0.373625</td>
      <td>0.885</td>
      <td>666.0</td>
      <td>11.078078</td>
      <td>2.977596</td>
      <td>5.0</td>
      <td>9.0</td>
      <td>10.0</td>
      <td>12.0</td>
      <td>27.0</td>
    </tr>
    <tr>
      <th>I</th>
      <td>654.0</td>
      <td>0.430979</td>
      <td>0.107285</td>
      <td>0.130</td>
      <td>0.3600</td>
      <td>0.440</td>
      <td>0.510</td>
      <td>0.725</td>
      <td>654.0</td>
      <td>0.329320</td>
      <td>...</td>
      <td>0.181875</td>
      <td>0.655</td>
      <td>654.0</td>
      <td>7.981651</td>
      <td>2.526363</td>
      <td>3.0</td>
      <td>6.0</td>
      <td>8.0</td>
      <td>9.0</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>M</th>
      <td>760.0</td>
      <td>0.562546</td>
      <td>0.103884</td>
      <td>0.000</td>
      <td>0.5100</td>
      <td>0.580</td>
      <td>0.630</td>
      <td>0.780</td>
      <td>760.0</td>
      <td>0.440559</td>
      <td>...</td>
      <td>0.360000</td>
      <td>0.885</td>
      <td>760.0</td>
      <td>10.731579</td>
      <td>3.071232</td>
      <td>0.0</td>
      <td>9.0</td>
      <td>10.0</td>
      <td>12.0</td>
      <td>27.0</td>
    </tr>
  </tbody>
</table>
<p>6 rows × 56 columns</p>
</div>




```python
# Save as CSV file
Descriptive_Combined_data.to_csv('Descriptive_Combined_data.csv', index=False)
```

Outliers


```python
# Determination  of outliers

def outlier_analysis(df):
    feature = 'Length(mm)'  
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identify and return outliers
    outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]
    return outliers

# Apply the function to the whole DataFrame
outliers = outlier_analysis(Combined_data)

```


```python
print(outliers)
# Save as CSV file
outliers.to_csv('outliers_LDH.csv', index=False)
```

Relationship between the variable using distribution features and correlational heatmap


```python
# Visualize outliers in Length, Diameter, and Height
plt.figure(figsize=(12, 6))
sns.boxplot(data=Combined_data[['Length(mm)', 'Diameter(mm)', 'Height(mm)']])
plt.title('Distribution and Outliers in Physical Dimensions of Abalones')
plt.ylabel('Measurements (mm)')
plt.show()
```


    
![png](output_21_0.png)
    



```python

```


```python
import seaborn as sns
import matplotlib.pyplot as plt

# Visualizing the data distribution by year 
sns.boxplot(x='Year', y='WholeWeight(g)', hue='Sex', data=Combined_data)
plt.title('Comparative Weights by Year and Sex')
plt.show()

```


    
![png](output_23_0.png)
    



```python
# Visualize data distributions by year
sns.boxplot(x='Year', y='ShuckedWeight(g)', hue='Sex', data=Combined_data)
plt.title('Comparative ShuckedWeight by Year and Sex')
plt.show()

```


    
![png](output_24_0.png)
    



```python
# Visualize data distributions by year
sns.boxplot(x='Year', y='SellWeight(g)', hue='Sex', data=Combined_data)
plt.title('Comparative SellWeight by Year and Sex')
plt.show()
```


    
![png](output_25_0.png)
    



```python
# Visualize data distributions by year
sns.boxplot(x='Year', y='Spots', hue='Sex', data=Combined_data)
plt.title('Comparative Spots by Year and Sex')
plt.show()

```


    
![png](output_26_0.png)
    



```python
# General distribution of features
sns.pairplot(Combined_data, hue='Sex')
plt.show()
# Save the figure

```


    
![png](output_27_0.png)
    



```python
# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(Combined_data.corr(), annot=True, cmap='coolwarm')
plt.show()
```


    
![png](output_28_0.png)
    


Model Building and Evaluation


```python
# setting the up model by first of all seprating into the health pedictor "Spot"

# Separate features and target
X = Combined_data.drop('Spots', axis=1)  # Assume 'Spots' is the target variable
y = Combined_data['Spots']

```


```python
# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

```


```python
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

```


```python
# Initialization of  models
Linear_regression_model = LinearRegression()
RandomForest_Model = RandomForestRegressor(n_estimators=100, random_state=42)
radientBoosting_Model = GradientBoostingRegressor(n_estimators=100, random_state=42)

```


```python
# Fitting models
Linear_regression_model.fit(X_train, y_train)
RandomForest_Model.fit(X_train, y_train)
radientBoosting_Model.fit(X_train, y_train)

```




<style>#sk-container-id-1 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: black;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-1 {
  color: var(--sklearn-color-text);
}

#sk-container-id-1 pre {
  padding: 0;
}

#sk-container-id-1 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-1 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-1 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-1 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-1 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-1 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-1 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-1 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-1 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-1 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-1 label.sk-toggleable__label {
  cursor: pointer;
  display: block;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
}

#sk-container-id-1 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-1 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-1 div.sk-label label.sk-toggleable__label,
#sk-container-id-1 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-1 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-1 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-1 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-1 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-1 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 1ex;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-1 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-1 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-1 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-1 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GradientBoostingRegressor(random_state=42)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;&nbsp;GradientBoostingRegressor<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.4/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html">?<span>Documentation for GradientBoostingRegressor</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>GradientBoostingRegressor(random_state=42)</pre></div> </div></div></div></div>




```python
Linear_regression_model.fit(X_train, y_train)
print("Linear Regression model trained.")
RandomForest_Model.fit(X_train, y_train)
print("Random Forest model trained.")
radientBoosting_Model.fit(X_train, y_train)
print("Gradient Boosting model trained.")

```

    Linear Regression model trained.
    Random Forest model trained.
    Gradient Boosting model trained.
    


```python
# Prediction on the test set
predictions_Linear_regression = Linear_regression_model.predict(X_test)
RandomForest = RandomForest_Model.predict(X_test)
radientBoosting = radientBoosting_Model.predict(X_test)

```


```python
from sklearn.metrics import mean_squared_error, r2_score

# Determination of  MSE and R² for each model
mse_Linear_Model = mean_squared_error(y_test, predictions_Linear_regression)
mse_Randorm_Forest = mean_squared_error(y_test, RandomForest)
mse_Gradient_Boosting = mean_squared_error(y_test, radientBoosting)

r2_Linear_Model = r2_score(y_test, predictions_Linear_regression)
r2_Randorm_Forest = r2_score(y_test, RandomForest)
r2_Gradient_Boosting = r2_score(y_test, radientBoosting)

print("Linear Regression - MSE:", mse_Linear_Model, "R²:", r2_Linear_Model)
print("Random Forest - MSE:", mse_Randorm_Forest, "R²:", r2_Randorm_Forest)
print("Gradient Boosting - MSE:", mse_Gradient_Boosting, "R²:", r2_Gradient_Boosting)

```

    Linear Regression - MSE: 5.700474623815357 R²: 0.47883017974348097
    Random Forest - MSE: 5.451251794258372 R²: 0.5016155486566927
    Gradient Boosting - MSE: 5.139953985105722 R²: 0.5300761653507019
    


```python
# feature importance for Random Forest and Gradient Boosting
importances_rf = model_rf.feature_importances_
importances_gb = model_gb.feature_importances_

# Use coefficients from Linear Regression as importance
importances_lr = abs(model_lr.coef_)  # Taking absolute value to reflect importance

# Creating a DataFrame for visualization from above 
features = X_train.columns
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance_LR': importances_lr,
    'Importance_RF': importances_rf,
    'Importance_GB': importances_gb
}).sort_values(by='Importance_RF', ascending=False)  # Sorting might vary depending on which model you prioritize

# Plotting feature importance
plt.figure(figsize=(14, 10))
plt.barh(importance_df['Feature'], importance_df['Importance_LR'], color='red', label='Linear Regression')
plt.barh(importance_df['Feature'], importance_df['Importance_RF'], color='skyblue', label='Random Forest', alpha=0.6)
plt.barh(importance_df['Feature'], importance_df['Importance_GB'], color='green', label='Gradient Boosting', alpha=0.6)
plt.xlabel('Feature Importance Score')
plt.title('Feature Importance for Predicting Abalone Health')
plt.legend()
plt.show()

```


```python
# End of code
```

```python

```  



