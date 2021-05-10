project_root = '/home/mccc/program/3DSSD-pytorch/'
# dataset
DATASET = dict(
    type='KittiDataset',
    dataset_cfg=dict(
        POINT_CLOUD_RANGE=[0., -40., -3., 70.4, 40., 1.],
        DATA_SPLIT=dict(train='train', test='val'),
        INFO_PATH=dict(train=['kitti_infos_train.pkl'], test=['kitti_infos_val.pkl']),
        FOV_POINTS_ONLY=True,
        DATA_AUGMENTOR=dict(
            AUG_METHOD_LIST=['gt_sampling', 'random_flip', 'random_rotation', 'random_scaling'],
            gt_sampling=dict(
                type='gt_sampling',
                configs=dict(
                    USE_ROAD_PLANE=True,
                    DB_INFO_PATH=['kitti_dbinfos_train.pkl'],
                    PREPARE=dict(
                        filter_by_min_points=['Car:5'],
                        filter_by_difficulty=[-1],
                    ),
                    SAMPLE_GROUPS=['Car:20'],
                    NUM_POINT_FEATURES=4,
                    GT_AUG_HARD_RATIO=0.6,
                    SAMPLE_WITH_FIX_NUM=False,
                    DATABASE_WITH_FAKELIDAR=False,
                    REMOVE_EXTRA_WIDTH=[0., 0., 0.],
                    LIMIT_WHOLE_SCENE=True,
                ),
            ),
            random_flip=dict(
                type='random_world_flip',
                configs=dict(
                    ALONG_AXIS_LIST=['x'],
                ),
            ),
            random_rotation=dict(
                type='random_world_rotation',
                configs=dict(
                    WORLD_ROT_ANGLE=[-0.78539816, 0.78539816],
                ),
            ),
            random_scaling=dict(
                type='random_world_scaling',
                configs=dict(
                    WORLD_SCALE_RANGE=[0.95, 1.05],
                ),
            ),
        ),
        DATA_PROCESSOR=dict(
            DATA_PROCESSOR_LIST=['mask_points', 'random_select', 'shuffle_points'],
            mask_points=dict(
                type='mask_points_and_boxes_outside_range',
                configs=dict(
                    REMOVE_OUTSIDE_BOXES=True,
                ),
            ),
            random_select=dict(
                type='random_select_points',
                configs=dict(
                    POINTS_NUM=dict(
                        train=16384,
                        test=32768,
                    ),
                    FAR_THRE=40.0,
                ),
            ),
            shuffle_points=dict(
                type='shuffle_points',
                configs=dict(
                    SHUFFLE_ENABLED=dict(
                        train=True,
                        test=False,
                    ),
                ),
            ),
        ),
    ),
    class_names=['Car'],
    training=True,
    root_path=project_root + 'data/KITTI',
    logger=None,
)

Model = dict(
    type='SingleStageDetector',
    backbone=dict(
        type='SingleStageBackbone',
        INPUT_FEATURE_DIM=1,
        Architecture=[
            [0, 0, [0.2, 0.4, 0.8], [32, 32, 64], [[16, 16, 32], [16, 16, 32], [32, 32, 64]], True,
             'D-FPS', 4096, 'layer1', 'BASE', 64, True],
            [1, 1, [0.4, 0.8, 1.6], [32, 32, 64], [[64, 64, 128], [64, 64, 128], [64, 96, 128]], True,
             'F-FPS', 512, 'layer2_1', 'BASE', 128, True],
            [1, 1, [0.4, 0.8, 1.6], [32, 32, 64], [[64, 64, 128], [64, 64, 128], [64, 96, 128]], True,
             'D-FPS', 512, 'layer2_2', 'BASE', 128, True],
            [2, 2, [1.6, 3.2, 4.8], [32, 32, 32], [[128, 128, 256], [128, 192, 256], [128, 256, 256]], True,
             'F-FPS', 256, 'layer3_1', 'BASE', 256, True],
            [3, 3, [1.6, 3.2, 4.8], [32, 32, 32], [[128, 128, 256], [128, 192, 256], [128, 256, 256]], True,
             'D-FPS', 256, 'layer3_2', 'BASE', 256, True],
        ],
        Loss_Config=dict(
            BACKBONE_LOSS=False,
            BACKBONE_LOSS_WEIGHT=[0., 1., 1., 1.],
        ),
    ),
    neck=dict(
        type='CandidateGeneration',
        Vote_Architecture=[4, 4, 128, True],
        CG_Architecture=[
            [[4, 5], [4, 5], [4.8, 6.4], [16, 32], [[256, 256, 512], [256, 512, 1024]],
             True, 'D-FPS', 'BASE', 512, True],
        ],
        MAX_TRANSLATE_RANGE=[-3.0, -2.0, -3.0],
        Loss_Config=dict(
            Vote_Loss=True,
            Vote_Loss_Weight=1.,
        ),
    ),
    head=dict(
        type='RegressionHead',
        Architecture=[[128, ], True],
        HEADING_BIN_NUM=12,
        IoU_HEAD=False,
        GT_EXTRA_WIDTH=[0.2, 0.2, 0.2],
        Loss_Config=dict(
            CORNER_LOSS=True,
            CLS_LOSS_WEIGHT=1.,
            IOU_LOSS_WEIGHT=1.,
            CORNER_LOSS_WEIGHT=1.,
            REGRESSION_LOSS_WEIGHTING=1.,
        ),
    ),
    iouhead=dict(
        type='IoUHead',
        CGE_Config=dict(
            MLP=[256, 256, 128, 128],
        ),
        MERGE_Config=dict(
            MLP=[128, 128],
        ),
    ),
    post_process=dict(
        CLS_THRESH=0.5,
        NMS_THRESH=0.1,
    ),
)

train = dict(
    work_dir=project_root + "train_info",
    log_dir=None,
    device_ids=[0, 1],
    checkpoint_path=None,
    OPTIMIZATION=dict(
        type='adam_onecycle',
        LR=0.004,
        WEIGHT_DECAY=0,
        MOMENTUM=0.9,
        MOMS=[0.95, 0.85],
        PCT_START=0.4,
        DIV_FACTOR=10,
        DECAY_STEP_LIST=[80, 120],
        LR_DECAY=0.1,
        LR_CLIP=0.0000001,
        LR_WARMUP=False,
        WARMUP_EPOCH=1,
        GRAD_NORM_CLIP=10,
    ),
    CONFIG=dict(
        MAX_ITERATION=100,
        BATCH_SIZE=8,
        EPOCH_RECORD_POINT=80,
        CHECKPOINT_INTERVAL=2,
        NUM_THREADS=8,
        LAUNCHER='none',
    ),
)

eval = dict(
    output_dir=project_root + 'data/KITTI/res/eval',
    work_dir=project_root + 'eval_info',
    checkpoint_path=project_root + 'train_info/20210318-033334/ckpt/checkpoint_100.pth',
    CONFIG=dict(
        BATCH_SIZE=40,
        NUM_THREADS=8,
    ),
)
