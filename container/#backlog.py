#backlog


def generate_figure(all_tables, repetition_times, signal, output_dir):
    global_data = {}  # Dictionnaire global pour contenir toutes les informations
    motion_outliers_list = [] 


    tasks = set()
    for table in all_tables:
        file_info = extract_file_info(os.path.basename(table).split('.')[0])
        tasks.add(file_info.get('task', 'N/A'))
    
    for table, repetition_time in zip(all_tables, repetition_times):
        # Extraire les informations du fichier
        file_info = extract_file_info(os.path.basename(table).split('.')[0])
        
        # Filtrer par tâche
        if file_info.get('task') != task:
            continue
        
        # Lire le tableau
        df = pd.read_csv(table, sep='\t')
        
        # Vérifier si le signal est dans le tableau
        if signal in df.columns:
            subject_name = os.path.basename(table).split('_')[0]
            task_name = file_info['task']
            session = os.path.basename(table).split('_')[1]
            run = os.path.basename(table).split('_')[3]
            
            # Initialiser le sujet s'il n'existe pas déjà
            if subject_name not in global_data:
                global_data[subject_name] = {'sessions': {}}
            
            # Initialiser la session s'il n'existe pas déjà
            if session not in global_data[subject_name]['sessions']:
                global_data[subject_name]['sessions'][session] = {'tasks': {}}
                
            # Initialiser la tâche s'il n'existe pas déjà
            if task_name not in global_data[subject_name]['sessions'][session]['tasks']:
                global_data[subject_name]['sessions'][session]['tasks'][task_name] = {'runs': {}}
            
            # Initialiser le run s'il n'existe pas déjà
            if run not in global_data[subject_name]['sessions'][session]['tasks'][task_name]['runs']:
                global_data[subject_name]['sessions'][session]['tasks'][task_name]['runs'][run] = []
            
            # Obtenir les données du signal
            signal_values = df[signal]
            time_indices = np.arange(0, len(signal_values) * repetition_time, repetition_time)


            motion_outliers_columns = [col for col in df.columns if 'motion_outlier' in col]


            # Somme logique le long de l'axe des colonnes
            df['motion_outliers_combined'] = df[motion_outliers_columns].sum(axis=1)

            # Convertir toute valeur >1 à 1
            motion_outliers_combined_binary= df['motion_outliers_combined'] = df['motion_outliers_combined'].apply(lambda x: 1 if x >= 1 else 0)

            motion_outliers_list = motion_outliers_combined_binary.tolist()
            motion_outliers_list = [int(item) for item in motion_outliers_list]

            #pd.set_option('display.max_rows', None)
            
            # Ajouter ces données au dictionnaire global
            global_data[subject_name]['sessions'][session]['tasks'][task_name]['runs'][run].append((table, time_indices, signal_values))

       ########################################################################################   
         
    fig = go.Figure()
    fig_tasks = go.Figure()

    visibility_by_subject = []

    for subject, subject_info in global_data.items():
        visibility = [False] * len(all_tables)

        for session, session_info in subject_info['sessions'].items():
            for task, task_info in session_info['tasks'].items():
                for run, data_list in task_info['runs'].items():
                    for current_table, time_indices, signal_values in data_list:
                        visibility[current_table] = True
                        custom_legend = f"{subject}_{session}_task-{task}_{run}"
                        new_trace = go.Scatter(x=time_indices, y=signal_values, mode='lines', name=custom_legend)

                        # Ajouter la trace et sauvegarder la couleur
                        fig.add_trace(new_trace)
                        fig_tasks.add_trace(new_trace)
                        # trace_colors[custom_legend] = new_trace.line.color

                        # print (trace_colors[custom_legend])

                        # for i, outlier in enumerate(motion_outliers_list):
                        #     if outlier == 1:
                        #         # Utiliser la couleur de la trace actuelle pour les outliers
                        #         outlier_color = trace_colors.get(custom_legend)

                        #         shape = go.layout.Shape(
                        #             type="line",
                        #             x0=i * repetition_time,
                        #             x1=i * repetition_time,
                        #             y0=0,
                        #             y1=1,
                        #             yref="paper",
                        #             line=dict(color=outlier_color, width=2)
                        #         )
                        #         fig.add_shape(shape)
                        #         fig_tasks.add_shape(shape)
                    visibility_by_subject.append(visibility)


########################################################################################    

    
    fig_tasks.update_layout(
        title={
            'text': f'{signal} for {task}',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        title_font=dict(size=22, color='rgb(107, 107, 107)', family="Georgia, serif"),
        xaxis_title='Time (seconds)',
        yaxis_title=f'{signal}',
        autosize=True
    )

    fig_tasks.update_layout(
        sliders=[
            {
                'active': 0,
                'yanchor': 'top',
                'xanchor': 'left',
                'currentvalue': {
                    'font': {'size': 20},
                    'prefix': 'Threshold:',
                    'visible': True,
                    'xanchor': 'right'
                },
                'transition': {'duration': 300, 'easing': 'cubic-in-out'},
                'pad': {'b': 10, 't': 50},
                'len': 0.9,
                'x': 0.1,
                'y': 0,
                'steps': [{
                    'args': [
                        {'frame': {'duration': 300, 'redraw': True}},
                        {'mode': 'immediate', 'transition': {'duration': 300}}
                    ],
                    'label': f"{round(threshold, 1)}",
                    'method': 'animate',
                    'value': f"{round(threshold, 1)}"
                } for threshold in np.arange(0, 2, 0.1)]
            }
        ],
    )


    fig.update_layout(
        title={
            'text': f'{signal} for all tasks',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        title_font=dict(size=22, color='rgb(107, 107, 107)', family="Georgia, serif"),
        xaxis_title='Time (seconds)', 
        yaxis_title=f'{signal}', 
        autosize=True
    )

    dropdown_buttons_all = []
    dropdown_buttons_tasks = []


    # Dropdown menu
    dropdown_buttons_all = [dict(label="All", method='update', args=[{'visible': [True]*len(fig.data)}, {'title': f'{signal} for All Subjects in All Tasks', 'showlegend': True}])]
    
    for i, (subject, _) in enumerate(global_data.items()):
            dropdown_buttons_all.append(dict(label=subject, method='update', args=[{'visible': visibility_by_subject[i]}, {'title': f'{signal} for {subject} in in All Tasks', 'showlegend': True}]))
    
    #for subject in visibility_by_subject.keys():
    #    dropdown_buttons_all.append(dict(label=subject, method='update', args=[{'visible': visibility_by_subject[subject]}, {'title': f'{signal} for {subject} in All Tasks', 'showlegend': True}]))
    
    dropdown_buttons_tasks = [dict(label="All", method='update', args=[{'visible': [True]*len(fig_tasks.data)}, {'title': f'{signal} for All Subjects in {task}', 'showlegend': True}])]
    for i, (subject, _) in enumerate(global_data.items()):
            dropdown_buttons_tasks.append(dict(label=subject, method='update', args=[{'visible': visibility_by_subject[i]}, {'title': f'{signal} for {subject} in task {task}', 'showlegend': True}]))

    fig.update_layout(hoverlabel_namelength=-1, updatemenus=[dict(active=0, buttons=dropdown_buttons_all, direction="down", pad={"r": 10, "t": 10}, showactive=True, x=0.1, xanchor="left", y=1.1, yanchor="top")])
    fig_tasks.update_layout(hoverlabel_namelength=-1, updatemenus=[dict(active=0, buttons=dropdown_buttons_tasks, direction="down", pad={"r": 10, "t": 10}, showactive=True, x=0.1, xanchor="left", y=1.1, yanchor="top")])

    if not all_tables:
        raise ValueError("all_tables must contain at least one element.")
    
  ########################################################################################   
    if not os.path.exists(output_dir):
            os.makedirs(output_dir)



    fig_name = f"desc-{signal}_signal_for_task-{task}.html"
    fig.write_html(os.path.join(output_dir, fig_name))
        
        # Si vous souhaitez sauvegarder fig_tasks
    fig_tasks_name = f"desc-{signal}_signal_tasks_for_task-{task}.html"
    fig_tasks.write_html(os.path.join(output_dir, fig_tasks_name))

    return tasks

    


    # for task in tasks:

    #     global_data, motion_outliers_list, repetition_time = read_and_preprocess_data(task, all_tables, repetition_times, signal)
        

        
    #     # Appeler plot_trace_data pour fig et obtenir visibility_by_subject
    #     visibility_by_subject = plot_trace_data(fig, fig_tasks, global_data, motion_outliers_list, repetition_time, all_tables)

    #     # Pas besoin de calculer visibility_lists manuellement car elles sont déjà calculées dans plot_trace_data
    #     # Vous pouvez donc simplement passer visibility_by_subject à configure_layout_and_interactivity
    #     configure_layout_and_interactivity(fig, fig_tasks, task, signal, visibility_by_subject, global_data)
        
    #     fig_name = f"desc-{signal}_signal_for_task-{task}.html"
    #     fig.write_html(os.path.join(output_dir, fig_name))
        
    #     # Si vous souhaitez sauvegarder fig_tasks
    #     fig_tasks_name = f"desc-{signal}_signal_tasks_for_task-{task}.html"
    #     fig_tasks.write_html(os.path.join(output_dir, fig_tasks_name))

    # return tasks  # si vous souhaitez renvoyer la liste des tâches uniques