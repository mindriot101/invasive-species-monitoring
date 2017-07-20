from currentmodel import build_model
import numpy as np
import pandas as pd

model = build_model()
model.load_weights('weights.h5')

X_test = np.load('test.npy')

predictions = model.predict(X_test)
predictions = predictions.ravel()
assert predictions.shape[0] == 1531

df = pd.DataFrame({'name': np.arange(len(predictions)) + 1, 
                   'invasive': predictions})
df[['name', 'invasive']].to_csv('submission.csv', index=False)
