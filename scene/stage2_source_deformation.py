"""可选静态来源相机；仅使用优化集相机，不访问GT标签。"""


def find_static_anchor_camera(args, train_cameras, key_function):
    mode = getattr(args, 'geometry_source_deformation_mode', 'dynamic')
    if mode == 'dynamic':
        return None
    if mode != 'anchor_camera':
        raise ValueError('未知来源形变模式：' + str(mode))
    stride = getattr(args, 'geometry_validation_stride', 10)
    if stride < 2:
        raise ValueError('静态来源相机要求几何验证间隔至少2。')
    key = getattr(args, 'geometry_static_anchor_key', '0002')
    optimization = {key_function(camera.image_name_train_light): camera
                    for i, camera in enumerate(train_cameras) if i % stride != 0}
    if key not in optimization:
        raise ValueError('静态来源相机必须属于优化集，不能使用验证或测试相机：' + str(key))
    return optimization[key]
