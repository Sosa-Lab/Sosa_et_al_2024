# Dictionaries of session meta-data per animal
import numpy as np

# Enter ppilot metadata starting with main task day 1
# Leave 'scan' as np.nan (not a number)
sosalab = {

    'pp01': ({'date': '2026_06_03', 'scene': 'FiveTower_Stay', 'session': 1, 'scan': np.nan, 'exp_day': 1,
                'GD': np.nan, 'pregnant': False, 'rig': 'omen-vr', 'reward_0': 'A', 'reward_1': 'A'},
             {'date': '2026_06_04', 'scene': 'FiveTower_Stay', 'session': 1, 'scan': np.nan, 'exp_day': 2, 
                'GD': np.nan, 'pregnant': False, 'rig': 'omen-vr', 'reward_0': 'A', 'reward_1': 'A'},
             {'date': '2026_06_05', 'scene': 'FiveTower_Switch_BlackoutDelay', 'session': 2, 'scan': np.nan, 'exp_day': 3, 
                'GD': np.nan, 'pregnant': False, 'rig': 'omen-vr', 'reward_0': 'A', 'reward_1': 'B'},
            {'date': '2026_06_06', 'scene': 'FiveTower_Stay_BlackoutDelay', 'session': 1, 'scan': np.nan, 'exp_day': 4, 
                'GD': np.nan, 'pregnant': False, 'rig': 'omen-vr', 'reward_0': 'B', 'reward_1': 'B'},
                ## Add remaining days below here!
    ),

    ## Add metadata for pp03 and ps04 using the same format as above
    'pp03': (

    ),

    'ps04': (

    )

}

## No need to edit below here

stanford = {
    # Behavior only animals for pregnancy pilot with RunningTraining -- here exp_day corresponds to GD

    '1780_02': ({'date': '25_05_2025', 'scene': 'RunningTraining_scan', 'session': 1, 'scan': np.nan, 'exp_day': 3, 'pregnant': True},
                {'date': '27_05_2025', 'scene': 'RunningTraining_scan',
                 'session': 1, 'scan': np.nan, 'exp_day': 5, 'pregnant': True},
                {'date': '29_05_2025', 'scene': 'RunningTraining_scan',
                 'session': 1, 'scan': np.nan, 'exp_day': 7, 'pregnant': True},
                {'date': '31_05_2025', 'scene': 'RunningTraining_scan',
                 'session': 1, 'scan': np.nan, 'exp_day': 9, 'pregnant': True},
                {'date': '02_06_2025', 'scene': 'RunningTraining_scan',
                 'session': 2, 'scan': np.nan, 'exp_day': 11, 'pregnant': True},
                {'date': '04_06_2025', 'scene': 'RunningTraining_scan',
                 'session': 1, 'scan': np.nan, 'exp_day': 13, 'pregnant': True},
                {'date': '06_06_2025', 'scene': 'RunningTraining_scan',
                    'session': 1, 'scan': np.nan, 'exp_day': 15, 'pregnant': True},
                ),

    '4855_04': ({'date': '31_05_2025', 'scene': 'RunningTraining_scan', 'session': 1, 'scan': np.nan, 'exp_day': 3, 'pregnant': False},
                {'date': '02_06_2025', 'scene': 'RunningTraining_scan',
                 'session': 1, 'scan': np.nan, 'exp_day': 5, 'pregnant': False},
                {'date': '04_06_2025', 'scene': 'RunningTraining_scan',
                 'session': 1, 'scan': np.nan, 'exp_day': 7, 'pregnant': False},
                {'date': '06_06_2025', 'scene': 'RunningTraining_scan',
                 'session': 1, 'scan': np.nan, 'exp_day': 9, 'pregnant': False},
                {'date': '08_06_2025', 'scene': 'RunningTraining_scan',
                 'session': 1, 'scan': np.nan, 'exp_day': 11, 'pregnant': False},
                {'date': '10_06_2025', 'scene': 'RunningTraining_scan',
                 'session': 1, 'scan': np.nan, 'exp_day': 13, 'pregnant': False},
                {'date': '12_06_2025', 'scene': 'RunningTraining_scan',
                 'session': 1, 'scan': np.nan, 'exp_day': 15, 'pregnant': False},
                ),

    '1781_06': ({'date': '31_05_2025', 'scene': 'RunningTraining_scan', 'session': 1, 'scan': np.nan, 'exp_day': 3, 'pregnant': False},
                {'date': '02_06_2025', 'scene': 'RunningTraining_scan',
                 'session': 1, 'scan': np.nan, 'exp_day': 5, 'pregnant': False},
                {'date': '04_06_2025', 'scene': 'RunningTraining_scan',
                 'session': 1, 'scan': np.nan, 'exp_day': 7, 'pregnant': False},
                {'date': '06_06_2025', 'scene': 'RunningTraining_scan',
                 'session': 1, 'scan': np.nan, 'exp_day': 9, 'pregnant': False},
                {'date': '08_06_2025', 'scene': 'RunningTraining_scan',
                 'session': 1, 'scan': np.nan, 'exp_day': 11, 'pregnant': False},
                {'date': '10_06_2025', 'scene': 'RunningTraining_scan',
                 'session': 1, 'scan': np.nan, 'exp_day': 13, 'pregnant': False},
                {'date': '12_06_2025', 'scene': 'RunningTraining_scan',
                 'session': 1, 'scan': np.nan, 'exp_day': 15, 'pregnant': False},
                ),

    '1781_07': ({'date': '29_05_2025', 'scene': 'RunningTraining_scan', 'session': 1, 'scan': np.nan, 'exp_day': 3, 'pregnant': True},
                {'date': '31_05_2025', 'scene': 'RunningTraining_scan',
                 'session': 1, 'scan': np.nan, 'exp_day': 5, 'pregnant': True},
                {'date': '02_06_2025', 'scene': 'RunningTraining_scan',
                 'session': 1, 'scan': np.nan, 'exp_day': 7, 'pregnant': True},
                {'date': '04_06_2025', 'scene': 'RunningTraining_scan',
                 'session': 1, 'scan': np.nan, 'exp_day': 9, 'pregnant': True},
                {'date': '06_06_2025', 'scene': 'RunningTraining_scan',
                'session': 1, 'scan': np.nan, 'exp_day': 11, 'pregnant': True},
                {'date': '08_06_2025', 'scene': 'RunningTraining_scan',
                'session': 1, 'scan': np.nan, 'exp_day': 13, 'pregnant': True},
                {'date': '10_06_2025', 'scene': 'RunningTraining_scan',
                 'session': 1, 'scan': np.nan, 'exp_day': 15, 'pregnant': True},
                ),
}
