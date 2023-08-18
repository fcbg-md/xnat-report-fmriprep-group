# def generate_figures(all_tables, repetition_times, signal, output_dir):

#     fig=go.Figure()

#     file_info_list = []

#     repetition_times_dict = {}  # New dictionary to store repetition times for each df

#     subjects_data = {}  # New dictionary to store data for each subject  

#     for table, repetition_time in zip(all_tables, repetition_times):
#         df = pd.read_csv(table, sep='\t')


        
#         # check if the signal is in the dataframe
#         if signal in df.columns:
#             df_signal = df[[signal]]
#             base_name = os.path.basename(table)

#             file_info = extract_file_info(os.path.basename(table).split('.')[0])

#             session_name = file_info['session']
#             task_name = file_info['task']
#             run_name = file_info['run']
#             subject_name = file_info['subject']

#             match = re.match(r"(sub-\w+)_ses-\w+_task-(\w+)_run-\w+_desc-\w+\.tsv", base_name)

#             if match:
#                 subject = match.group(1)
#                 task = match.group(2)
#                 file_num = all_tables.index(table) + 1
#                 file_info_list.append([file_num, base_name, subject, task])

#             if subject_name not in subjects_data:
#                 subjects_data[subject_name] = {}
#             if session_name not in subjects_data[subject_name]:
#                 subjects_data[subject_name][session_name] = []

#             fig.add_trace(go.Scatter(x=time_indices, y=signal_values, mode='lines', name=custom_legend)) 

            #print(f"Repetition time for {custom_legend} is {repetition_time} seconds.")

#     # Convert the list of lists into a dataframe
#     columns = ["file_num", "file_name", "subject", "task"]
#     df = pd.DataFrame(file_info_list, columns=columns)


#     df_input = df.copy()

# # split df by labels
#     subjects = df['subject'].unique().tolist()
#     file_num = df['file_num'].unique().tolist()  

#     # dataframe collection grouped by labels
#     dfs = {}
#     for subject in subjects:
#         dfs[subject]=pd.pivot_table(df[df['subject']==subject],
#                                         values='file_name',
#                                         index=['file_num'],
#                                         columns=['task'],
#                                         aggfunc=np.sum)

#     # find row and column unions
#     common_cols = []
#     common_rows = []
#     for df in dfs.keys():
#         common_cols = sorted(list(set().union(common_cols,list(dfs[df]))))
#         common_rows = sorted(list(set().union(common_rows,list(dfs[df].index))))

#     # find dimensionally common dataframe
#     df_common = pd.DataFrame(np.nan, index=common_rows, columns=common_cols)

#     # reshape each dfs[df] into common dimensions
#     dfc={}
#     for df_item in dfs:
#         #print(dfs[unshaped])
#         df1 = dfs[df_item].copy()
#         s=df_common.combine_first(df1)
#         df_reshaped = df1.reindex_like(s)
#         dfc[df_item]=df_reshaped

#     dropdown_buttons = []
    
#     # "All" option
#     dropdown_buttons.append(dict(label="All", method='update', args=[{'visible': [True]*len(fig.data)}, {'title': f'{signal} for All Subjects and Sessions', 'showlegend': True}]))
    
#     for subject in subjects_data:
#         for session in subjects_data[subject]:
#             label = f"{subject} - Session: {session}"
#             visibility = [trace.name.startswith(f"{subject}_ses-{session}") for trace in fig.data]
#             dropdown_buttons.append(dict(label=label, method='update', args=[{'visible': visibility}, {'title': f'{signal} for {label}', 'showlegend': True}]))
            

#     fig.update_layout(
#         updatemenus=[
#             dict(buttons=dropdown_buttons, direction="down", x=0.1, xanchor="left", y=1.1, yanchor="top")
#         ]
#     )

#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     fig_name = f"desc-{signal}_signal_for_all_subjects.html"
#     fig.write_html(os.path.join(output_dir, fig_name))

#     fig.show()
