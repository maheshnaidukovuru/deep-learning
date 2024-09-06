import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Change backend (if needed)
import matplotlib
matplotlib.use('TkAgg')  # Comment this line if it's not needed in your environment

# Create the NumPy array for actual and predicted labels.
actual = np.array(['Dog', 'Dog', 'Dog', 'Not Dog', 'Dog', 'Not Dog', 'Dog', 'Dog', 'Not Dog', 'Not Dog'])
predicted = np.array(['Dog', 'Not Dog', 'Dog', 'Not Dog', 'Dog', 'Dog', 'Dog', 'Dog', 'Not Dog', 'Not Dog'])

# Compute the confusion matrix.
cm = confusion_matrix(actual, predicted)

# Plot the confusion matrix.
sns.heatmap(cm, annot=True, fmt='g', xticklabels=['Dog', 'Not Dog'], yticklabels=['Dog', 'Not Dog'])
plt.ylabel('Prediction', fontsize=13)
plt.xlabel('Actual', fontsize=13)
plt.title('Confusion Matrix', fontsize=17)

# Save or show the plot.
plt.savefig('confusion_matrix.png')  # Save the plot as an image file
plt.show()  # Display the plot (if interactive backend is set)
