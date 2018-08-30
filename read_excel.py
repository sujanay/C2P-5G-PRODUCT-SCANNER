import pandas as pd
import numpy as np

filename = 'Neural_Network_Training_and_Testing_Parameters.xlsx'
df = pd.read_excel(filename)
columns = np.array(df.columns)

print(columns)

# # This program reads the excel file and converts every individual columns
# # into separate .txt file.
# import os
# import numpy as np
# import pandas as pd
#
# # Ask for the input file name
# filename = input("Enter the file name<eg. 'filename.xlsx'>:")
#
# # Read the excel file
# df = pd.read_excel(filename)  # Name of file you want to operate on
#
# # Identify the columns of the excel file
# columns = np.array(df.columns)
#
# # Start creating text file for each of the columns in the excel file
# print('Processing...')
# for individual_column in columns:
#     # Save each individual column in a separate text file without column name
#     df[individual_column].to_csv(individual_column + '.txt', index=False, mode='a')
#
#     # Let the user know about the creation of the files being processed
#     print('Text file for ' + individual_column + '.txt' + ' has been created')
#
# cwd = os.getcwd()
#
# # Completion of the process.
# print('Process Complete!\nPlease confirm the file creation at <' + cwd + '>')