(define (domain robocon-map)
    (:requirements :strips :typing :fluents) 

    ; --- 1. 类别 (Types) ---
    (:types
        location
        
        robot_friendly - robot
        robot_opponent - robot
        robot - object ; Base type for all robots
        
        item 
        kfs_r1 - item 
        kfs_r2 - item 
        kfs_fake - item 
        staff - item 
        spearhead - item 
        weapon - item 
    )

    ; --- 2. 状态谓词 (Predicates) ---
    (:predicates
        ; --- 地图拓扑与可达性 ---
        (connected ?from - location ?to - location)
        (is_accessible ?l - location) ; R2 可进入区域
        (is_blocked ?l - location)    ; 动态障碍 (R1, 敌方机器人)
        (is_forest_block ?l - location) 
        (is_start_block ?l - location) ; F1, F2, F3
        (is_exit_block ?l - location)  ; F10, F11, F12
        (adjacent ?l1 - location ?l2 - location) ; F区相邻关系

        ; --- 机器人/物品状态 ---
        (at ?r - robot ?l - location)     
        (item_at ?i - item ?l - location) 

        (gripper_empty ?r - robot)        
        (holding ?r - robot ?i - item)    

        ; --- 任务/规则状态 ---
        (r1_left_mc)                 ; Rule 4.3.10: R2 must wait for R1
        (r2_picked_first_kfs)        ; Rule 4.4.14: R2's first KFS must be from start block
    )

    ; --- 3. 动作 (Actions) ---

    ; --- 动作 1: 移动 (通用, 考虑障碍) ---
    (:action move
        :parameters (?r - robot_friendly ?from - location ?to - location)
        :precondition (and
            (at ?r ?from)
            (connected ?from ?to)
            (is_accessible ?to)
            (not (is_blocked ?to)) ; 避障判断
        )
        :effect (and
            (at ?r ?to)
            (not (at ?r ?from))
        )
    )
    
    ; --- 动作 1.1: R1 离开武馆 (触发 r1_left_mc) ---
    (:action r1_leave_mc
        :parameters (?r - robot_friendly ?from - location ?to - location ?w - weapon)
        :precondition (and
            (at ?r ?from)
            (connected ?from ?to)
            (is_accessible ?to)
            (not (is_blocked ?to))
            (holding ?r ?w)      ; R1 必须拿着兵器 (Rule 4.3.9)
        )
        :effect (and
            (at ?r ?to)
            (not (at ?r ?from))
            (r1_left_mc) 
        )
    )
    
    ; --- 动作 1.2: R2 离开武馆 (等待 r1_left_mc) ---
    (:action r2_leave_mc
        :parameters (?r - robot_friendly ?from - location ?to - location)
        :precondition (and
            (at ?r ?from)
            (connected ?from ?to)
            (is_accessible ?to)
            (not (is_blocked ?to))
            (r1_left_mc) ; R1 必须已经离开了 (Rule 4.3.10)
        )
        :effect (and
            (at ?r ?to)
            (not (at ?r ?from))
        )
    )

    ; --- 动作 2: 组装兵器 (R1+R2 协作) ---
    (:action assemble_weapon
        :parameters (?r1 - robot_friendly ?r2 - robot_friendly ?s - staff ?h - spearhead ?w - weapon ?l - location)
        :precondition (and
            (at ?r1 ?l) (at ?r2 ?l)             ; R1, R2 在同一地点
            (holding ?r1 ?s)                    ; R1 拿着矛杆
            (holding ?r2 ?h)                    ; R2 拿着矛头
            (item_at ?w limbo)                  ; 兵器在虚拟空间，等待被“创造”
        )
        :effect (and
            (not (holding ?r1 ?s)) (gripper_empty ?r1)
            (not (holding ?r2 ?h)) (gripper_empty ?r2)
            (not (item_at ?w limbo)) 
            (holding ?r1 ?w) ; R1 拿着组装好的兵器 (Rule 4.3.8)
        )
    )
    
    ; --- 动作 3: 捡起/放下 (通用) ---
    (:action pickup
        :parameters (?r - robot_friendly ?i - item ?l - location)
        :precondition (and (at ?r ?l) (item_at ?i ?l) (gripper_empty ?r))
        :effect (and (holding ?r ?i) (not (gripper_empty ?r)) (not (item_at ?i ?l)))
    )

    (:action drop
        :parameters (?r - robot_friendly ?i - item ?l - location)
        :precondition (and (at ?r ?l) (holding ?r ?i))
        :effect (and (item_at ?i ?l) (gripper_empty ?r) (not (holding ?r ?i)))
    )

    ; --- 动作 4: R2 KFS 特殊拾取规则 (Rule 4.4.14, 4.4.15) ---

    ; R2 捡第一个 KFS：必须在入口块 (F1, F2, F3)
    (:action r2_pickup_first_kfs
        :parameters (?r - robot_friendly ?k - kfs_r2 ?l - location)
        :precondition (and
            (at ?r ?l) (item_at ?k ?l) (gripper_empty ?r)
            (is_start_block ?l) 
            (not (r2_picked_first_kfs)) ; 必须是第一次捡
        )
        :effect (and
            (holding ?r ?k) (not (gripper_empty ?r)) (not (item_at ?k ?l))
            (r2_picked_first_kfs) 
        )
    )
    
    ; R2 捡后续 KFS：必须在相邻块
    (:action r2_pickup_adjacent_kfs
        :parameters (?r - robot_friendly ?k - kfs_r2 ?l_robot - location ?l_kfs - location)
        :precondition (and
            (at ?r ?l_robot) (item_at ?k ?l_kfs) (gripper_empty ?r)
            (r2_picked_first_kfs)         ; 必须已经捡过第一个
            (adjacent ?l_robot ?l_kfs)    ; 必须相邻
        )
        :effect (and
            (holding ?r ?k) (not (gripper_empty ?r)) (not (item_at ?k ?l_kfs))
        )
    )
)
