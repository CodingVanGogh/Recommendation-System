## Simple Recommendation System in Python3+ (Using Collaborative Filtering)

#### Installation
- pip install recommendation_system 

#### Example Usage
```python
from recommendation_system import complete_recommendation_table
import pandas as pd
Y_df = pd.DataFrame({'Bob': [5, '?', 4], 'Cathy': [5, 4, '?'], \
                    'Dave' : [2, 5, 5]}, index=['Toy Story',  \
                    'Despicble Me', 'Spiderman'])
output = complete_recommendation_table(Y_df, len(Y_df) + 1, \
            unknown='?', max_value=5, min_value=0, regularization_coeff=0.2)
print(output)
 ```

### Example Output
- Assume an input table that looks like this:
- |Movie /User&nbsp;&nbsp;&nbsp;| Bob   | Cathy | Dave |
- | Toy Story&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;| &nbsp;5&nbsp;&nbsp;&nbsp;&nbsp;| &nbsp;5&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;| &nbsp;2&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|
- | Despicble Me   | &nbsp;?&nbsp;&nbsp;&nbsp;&nbsp;| &nbsp;4&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;| &nbsp;5&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|
- | Spiderman&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;| &nbsp;4&nbsp;&nbsp;&nbsp;| &nbsp;?&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;| &nbsp;5&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|
- The output after filling in the '?' would look something like this:
- |Movie /User&nbsp;&nbsp;&nbsp;| Bob&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;| Cathy&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;| Dave&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|
- | Toy Story&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;| &nbsp;4.808920&nbsp;&nbsp;&nbsp;&nbsp;| &nbsp;4.917348&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;| &nbsp;2.118293&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|
- | Despicble Me   | &nbsp;3.998761&nbsp;&nbsp;&nbsp;&nbsp;| &nbsp;3.874179&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;| &nbsp;4.816824&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|
- | Spiderman&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;| &nbsp;3.818917&nbsp;&nbsp;&nbsp;| &nbsp;3.744278&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;| &nbsp;4.857076&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|
- The function 'complete_recommendation_table' fills all unknown values('?') with the predictions
