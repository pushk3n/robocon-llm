import json
import os
import collections

# --- 1. 路径配置 ---
# (请确保这三个文件与此脚本位于同一目录, 或者修改为正确的相对路径)
MAP_PATH = '../data/initial/map.json'
RULES_PATH = '../data/initial/map_rules_cn.json'
ADJCENT_PATH = '../data/initial/map_accessible_adjcent.json' # <-- 修改: 使用您修订后的邻接文件

OUTPUT_DIR = '../data/processed/'
OUTPUT_FILENAME_JSONL = 'training_dataset_v2.jsonl'
OUTPUT_FILENAME_JSON = 'training_dataset_v2.json'

# --- 2. 数据加载 ---

def load_data(map_path, rules_path, adjcent_path): # <-- 修改: 参数变更
    """加载所有三个JSON源文件"""
    print("正在加载源文件...")
    try:
        with open(map_path, 'r', encoding='utf-8') as f:
            map_data = json.load(f)
        with open(rules_path, 'r', encoding='utf-8') as f:
            rules_data = json.load(f)
        with open(adjcent_path, 'r', encoding='utf-8') as f: # <-- 修改: 加载新文件
            adjcent_data = json.load(f)
        print("所有文件加载成功!")
        return map_data, rules_data, adjcent_data # <-- 修改: 返回新数据
    except FileNotFoundError as e:
        print(f"错误: 缺少文件 {e.filename}")
        print("请确保 map.json, map_rules_cn.json, 和 map_accessible_adjcent.json 都在正确的路径中。") # <-- 修改: 更新提示
        return None, None, None

def build_helpers(map_data, adjcent_data): # <-- 修改: 参数变更
    """构建辅助数据结构以便快速查询"""
    
    # 1. zones_map: 按名称快速查找zone对象
    zones_map = {zone['name']: zone for zone in map_data['zones']}
    
    # 2. adj_list: 邻接表 (图)
    # (!!! 关键修改: 现在直接从 map_accessible_adjcent.json 读取 !!!)
    adj_list = collections.defaultdict(list)
    for item in adjcent_data:
        zone_name = item.get("name")
        if zone_name in zones_map: # 确保区域在 map.json 中存在
            adj_list[zone_name] = item.get("adjacent_zone", [])
    # (旧的 verges 逻辑已被移除)

    # 3. R2可访问性 (逻辑不变)
    r2_accessible = {
        zone['name']: zone.get('R2_access', map_data.get('default_R2_access', True))
        for zone in map_data['zones']
    }
            
    return zones_map, adj_list, r2_accessible

# --- 3. 路径搜索算法 ---

def bfs_shortest_path(adj, r2_accessible, start_node, end_node):
    """
    使用BFS查找R2可通行的最短路径
    (V2 - 已更新, 包含 Rule 8/14/15 的 ramp 访问限制)
    """
    if start_node not in r2_accessible or end_node not in r2_accessible:
        return "起始节点或结束节点不存在。"
        
    # 检查R2是否可以进入起点和终点
    if not r2_accessible.get(start_node, False): # 使用 .get() 增加安全性
        return f"R2机器人无法进入起始区域 {start_node}。"
    if not r2_accessible.get(end_node, False):
        return f"R2机器人无法进入目标区域 {end_node}。"

    queue = collections.deque([(start_node, [start_node])]) # (节点, 路径)
    visited = {start_node}

    while queue:
        current_node, path = queue.popleft()

        if current_node == end_node:
            return " -> ".join(path)

        for neighbor in adj.get(current_node, []): # 使用 .get() 增加安全性
            if neighbor not in visited:
                
                # --- [!! 关键规则检查 !!] ---
                # 规则 8/14/15/18/19: 检查 ramp 的特殊连接规则
                
                # 1. 检查是否 *进入* ramp
                if neighbor == "ramp" and current_node not in ["R2_EX_zone1", "ARENA_idle1"]:
                    continue # 非法进入 ramp, 跳过此路径
                
                # 2. 检查是否 *离开* ramp
                if current_node == "ramp" and neighbor not in ["R2_EX_zone1", "ARENA_idle1"]:
                    continue # 非法离开 ramp, 跳过此路径
                # --- [!! 规则检查结束 !!] ---

                # 关键: 检查R2是否可以访问邻居
                if r2_accessible.get(neighbor, False):
                    visited.add(neighbor)
                    new_path = list(path)
                    new_path.append(neighbor)
                    queue.append((neighbor, new_path))
    
    return f"未找到从 {start_node} 到 {end_node} 的R2可通行路径。"


# --- 4. 生成器 (所有生成器均无需修改) ---

def generate_coordinate_queries(zones):
    """生成坐标查询类数据"""
    dataset = []
    for zone in zones:
        dataset.append({
            "instruction": f"{zone['name']}区域的坐标范围是什么?",
            "input": "",
            "output": f"{zone['name']}区域的坐标范围是:x轴从{zone['x_min']}到{zone['x_max']},y轴从{zone['y_min']}到{zone['y_max']}。"
        })
        dataset.append({
            "instruction": f"告诉我{zone['name']}的位置。",
            "input": "",
            "output": f"{zone['name']}的位置是:x轴范围{zone['x_min']}-{zone['x_max']},y轴范围{zone['y_min']}-{zone['y_max']}。"
        })
    return dataset

def generate_access_queries(zones, default_access):
    """生成访问权限类数据"""
    dataset = []
    forbidden_zones = [z['name'] for z in zones if not z.get('R2_access', default_access)]
    accessible_zones = [z['name'] for z in zones if z.get('R2_access', default_access)]

    for zone in zones:
        r2_access = zone.get('R2_access', default_access)
        access_str = "可以" if r2_access else "不可以"
        dataset.append({
            "instruction": f"R2机器人能否进入{zone['name']}区域?",
            "input": "",
            "output": f"R2机器人{access_str}进入{zone['name']}区域。"
        })
    
    dataset.append({
        "instruction": "列出R2机器人不能进入的所有区域。",
        "input": "",
        "output": f"R2机器人不能进入的区域包括: {', '.join(forbidden_zones)}。"
    })
    dataset.append({
        "instruction": "R2机器人可以进入哪些区域?",
        "input": "",
        "output": f"R2机器人可以进入的区域有: {', '.join(accessible_zones)}。"
    })
    return dataset

def generate_point_queries(zones):
    """生成坐标点归属判断数据"""
    dataset = []
    for zone in zones:
        center_x = (zone['x_min'] + zone['x_max']) // 2
        center_y = (zone['y_min'] + zone['y_max']) // 2
        dataset.append({
            "instruction": f"坐标点({center_x}, {center_y})位于哪个区域?",
            "input": "",
            "output": f"坐标点({center_x}, {center_y})位于{zone['name']}区域。"
        })
    return dataset

def generate_hierarchy_queries(zones):
    """生成层级关系数据"""
    dataset = []
    parent_children = {}
    child_parent = {}
    
    for zone in zones:
        if 'parent_zone' in zone and zone['parent_zone']:
            child_parent[zone['name']] = zone['parent_zone']
        if 'sub_zones' in zone:
            parent_children[zone['name']] = zone['sub_zones']

    for parent, children in parent_children.items():
        dataset.append({
            "instruction": f"{parent}区域包含哪些子区域?",
            "input": "",
            "output": f"{parent}区域包含以下子区域: {', '.join(children)}。"
        })

    for child, parent in child_parent.items():
        dataset.append({
            "instruction": f"{child}区域的父区域是什么?",
            "input": "",
            "output": f"{child}区域的父区域是{parent}。"
        })
        
    # 多层级查询
    for child, parent in child_parent.items():
        if parent in child_parent:
            grandparent = child_parent[parent]
            dataset.append({
                "instruction": f"{child}区域的完整层级路径是什么?",
                "input": "",
                "output": f"{child}区域的完整层级路径是: {grandparent} -> {parent} -> {child}。"
            })
    return dataset

def generate_height_queries(zones):
    """生成高度信息数据"""
    dataset = []
    forest_zones = [z for z in zones if z['name'].startswith('F') and z['name'][1:].isdigit()]
    
    height_groups = {'low': [], 'mid': [], 'high': []}
    for zone in forest_zones:
        if 'height_level' in zone:
            dataset.append({
                "instruction": f"{zone['name']}区域的高度等级是多少?",
                "input": "",
                "output": f"{zone['name']}区域的高度等级是{zone['height_level']},属于{zone['high_type']}类型。"
            })
            if 'high_type' in zone:
                 height_groups[zone['high_type']].append(zone['name'])

    dataset.append({
        "instruction": "forest森林区域中,哪些格子是低地?",
        "input": "",
        "output": f"forest森林区域中,低地格子有: {', '.join(height_groups['low'])}。"
    })
    dataset.append({
        "instruction": "forest森林区域中,哪些格子是中地?",
        "input": "",
        "output": f"forest森林区域中,中地格子有: {', '.join(height_groups['mid'])}。"
    })
    dataset.append({
        "instruction": "forest森林区域中,哪些格子是高地?",
        "input": "",
        "output": f"forest森林区域中,高地格子有: {', '.join(height_groups['high'])}。"
    })
    return dataset

def generate_rule_queries(rules):
    """(新增) 生成地图规则数据"""
    dataset = []
    for rule in rules:
        dataset.append({
            "instruction": f"地图规则{rule['id']}是什么?",
            "input": "",
            "output": f"规则{rule['id']}是: {rule['text']}"
        })
    # 针对关键规则的特定提问
    dataset.append({
        "instruction": "R2机器人的默认访问权限是什么?",
        "input": "",
        "output": "未被标记为\"R2_access: false\"的区域默认为R2都可以进入, 但是明确标记了\"R2_access: false\"的区域R2均不可进入."
    })
    dataset.append({
        "instruction": "R2机器人如何从MF进入ARENA?",
        "input": "",
        "output": "根据规则(Rule 8), R2只能从R2_EX_zone1 (ID:13) 或 ARENA_idle1 (ID:36) 进入 ramp."
    })
    # (!!! 关键修改: 更新Rule 8的答案 !!!)
    dataset.append({
        "instruction": "R2机器人如何从MF区域进入ARENA的路径是什么?",
        "input": "",
        "output": "根据规则(Rule 8), R2必须从R2_EX_zone1开始, 经过ramp和ARENA_idle1, 最终到达retry_zone。"
    })
    # (!!! 关键修改: 增加Rule 15的测试 !!!)
    dataset.append({
        "instruction": "R2机器人是否可以从R2_EX_zone2进入ramp?",
        "input": "",
        "output": "不可以。根据规则(Rule 15), R2不可以从 R2_EX_zone2 或者 ARENA_idle3 进入 ramp。"
    })
    
    return dataset

def generate_topology_queries(adj_list, r2_accessible):
    """(新增) 生成拓扑邻接数据"""
    dataset = []
    for zone, neighbors in adj_list.items():
        if not neighbors: # 跳过没有邻居的区域 (如 'MC', 'MF' 等)
            continue
            
        dataset.append({
            "instruction": f"{zone}区域和哪些区域是相邻的?",
            "input": "",
            "output": f"{zone}区域与以下区域相邻: {', '.join(neighbors)}。"
        })
        
        # 增加R2可通行性判断
        for neighbor in neighbors:
            instr = f"R2机器人是否可以从{zone}移动到{neighbor}?"
            
            # 确保zone和neighbor都在r2_accessible字典中
            if zone not in r2_accessible or neighbor not in r2_accessible:
                out = f"不可以。区域 {zone} 或 {neighbor} 的访问权限未定义。"
            
            # [!! 关键修改: 增加 ramp 规则检查 !!]
            elif neighbor == "ramp" and zone not in ["R2_EX_zone1", "ARENA_idle1"]:
                out = f"不可以。根据规则(Rule 15), R2机器人不被允许从{zone}进入ramp。"
            elif zone == "ramp" and neighbor not in ["R2_EX_zone1", "ARENA_idle1"]:
                out = f"不可以。根据规则(Rule 15), R2机器人不被允许从ramp移动到{neighbor}。"
            # (Ramp规则检查结束)
                
            elif r2_accessible[zone] and r2_accessible[neighbor]:
                out = f"可以。{zone}和{neighbor}是相邻的, 并且R2机器人可以进入这两个区域。"
            elif not r2_accessible[zone]:
                out = f"不可以。R2机器人无法进入出发区域{zone}。"
            elif not r2_accessible[neighbor]:
                out = f"不可以。虽然{zone}和{neighbor}是相邻的, 但R2机器人被禁止进入{neighbor}区域。"
            else:
                 out = "不可以。机器人无法访问该路径。" # 兜底
            
            dataset.append({"instruction": instr, "input": "", "output": out})
    return dataset

def generate_pathfinding_queries(adj_list, r2_accessible, zones_map):
    """(新增) 生成路径推理数据 (V3 - 规则感知版)"""
    print("  -> 正在生成健壮的路径推理数据 (V3)...")
    dataset = []
    
    # 获取所有可访问和不可访问的区域
    accessible_zones = {z for z, can_access in r2_accessible.items() if can_access and z in adj_list}
    forbidden_zones = {z for z, can_access in r2_accessible.items() if not can_access}

    # --- 1. 关键成功路径 (Key Success Paths) ---
    key_paths = [
        ("R2_start_zone", "F8"),
        # ("R2_start_zone", "retry_zone"), # <-- 此路径现在更复杂, 放入规则组
        ("R2_EN_zone", "F12"),
        ("MC_idle1", "MC_idle5"),
        ("F1", "F12"),
        ("ARENA_idle1", "ARENA_idle5")
    ]
    
    for start, end in key_paths:
        path_result = bfs_shortest_path(adj_list, r2_accessible, start, end)
        dataset.append({
            "instruction": f"R2机器人从{start}到{end}的最短路径是什么?",
            "input": "",
            "output": f"从{start}到{end}的最短路径是: {path_result}。"
        })
        
    # --- 2. 强制规则路径 (Rule 8 / 14 / 15) ---
    
    # 2a. 测试 Rule 8/14: 合法进入 ramp (R2_EX_zone1 -> retry_zone)
    start, end = "R2_EX_zone1", "retry_zone"
    path_result = bfs_shortest_path(adj_list, r2_accessible, start, end)
    dataset.append({
        "instruction": f"R2机器人如何从{start}移动到{end}?",
        "input": "",
        "output": f"R2机器人必须按此路径移动: {path_result}。这符合Rule 8/14, 从R2_EX_zone1进入ramp。"
    })
    
    # 2b. 测试 Rule 8/14: 合法进入 ramp (ARENA_idle1 -> R2_EX_zone1)
    start, end = "ARENA_idle1", "R2_EX_zone1"
    path_result = bfs_shortest_path(adj_list, r2_accessible, start, end)
    dataset.append({
        "instruction": f"R2机器人如何从{start}移动到{end}?",
        "input": "",
        "output": f"R2机器人必须按此路径移动: {path_result}。这符合Rule 8/14, 从ARENA_idle1进入ramp。"
    })

    # 2c. 测试 Rule 15: 非法进入 ramp (假设 R2_EX_zone2 和 ramp 在 adjcent 中是邻居)
    # (注意: 我们的BFS现在会智能地绕开这条非法路径)
    start, end = "R2_EX_zone2", "retry_zone" 
    path_result = bfs_shortest_path(adj_list, r2_accessible, start, end) 
    dataset.append({
        "instruction": f"R2机器人能从{start}移动到{end}吗?",
        "input": "",
        "output": f"不可以。{path_result} (根据Rule 15, R2不可以从{start}进入ramp, 无法规划路径)。"
    })

    # 2d. 测试 Rule 15: 非法进入 ramp (假设 ARENA_idle3 和 ramp 在 adjcent 中是邻居)
    start, end = "ARENA_idle3", "R2_EX_zone1" 
    path_result = bfs_shortest_path(adj_list, r2_accessible, start, end)
    dataset.append({
        "instruction": f"R2机器人能从{start}移动到{end}吗?",
        "input": "",
        "output": f"不可以。{path_result} (根据Rule 15, R2不可以从{start}进入ramp, 无法规划路径)。"
    })

    # --- 3. 失败路径: 起点/终点不可访问 ---
    
    # 3a. 起点不可访问
    test_end_point = "R2_start_zone" # 一个已知的可访问点
    for start_zone in forbidden_zones:
        if start_zone not in adj_list: continue # 跳过没有连接的区域
        path_result = bfs_shortest_path(adj_list, r2_accessible, start_zone, test_end_point)
        dataset.append({
            "instruction": f"R2机器人从{start_zone}到{test_end_point}的路径是什么?",
            "input": "",
            "output": f"{path_result}" # 应该是 "R2机器人无法进入起始区域..."
        })

    # 3b. 终点不可访问
    test_start_point = "R2_start_zone" # 一个已知的可访问点
    for end_zone in forbidden_zones:
        if end_zone not in adj_list: continue
        path_result = bfs_shortest_path(adj_list, r2_accessible, test_start_point, end_zone)
        dataset.append({
            "instruction": f"R2机器人从{test_start_point}到{end_zone}的路径是什么?",
            "input": "",
            "output": f"{path_result}" # Gördüğünüz gibi "R2机器人无法进入目标区域..."
        })

    # --- 4. 失败路径: 路径被阻断 (R1_path_1, R1_path_2) ---
    start, end = "R1_start_zone", "R2_EX_zone1" # 此路径必须经过 R1_path_1 (R2_access: false)
    path_result = bfs_shortest_path(adj_list, r2_accessible, start, end)
    dataset.append({
        "instruction": f"R2机器人从{start}到{end}的最短路径是什么?",
        "input": "",
        "output": f"{path_result}" # 应该是 "未找到...R2可通行路径"
    })
    
    start, end = "MC_idle5", "R2_EX_zone2" # 此路径必须经过 R1_path_2 (R2_access: false)
    path_result = bfs_shortest_path(adj_list, r2_accessible, start, end)
    dataset.append({
        "instruction": f"R2机器人从{start}到{end}的最短路径是什么?",
        "input": "",
        "output": f"{path_result}" # 应该是 "未找到...R2可通行路径"
    })
    
    return dataset


# --- 5. 主函数 ---

def main():
    # <-- 修改: 调用新的加载函数
    map_data, rules_data, adjcent_data = load_data(MAP_PATH, RULES_PATH, ADJCENT_PATH)
    if not map_data:
        return

    zones = map_data['zones']
    default_access = map_data.get('default_R2_access', True)
    
    # 构建辅助工具
    # <-- 修改: 传入新的adjcent_data
    zones_map, adj_list, r2_accessible = build_helpers(map_data, adjcent_data)
    
    # 生成各类数据
    print("开始生成训练数据...")
    all_datasets = []
    
    # 基础知识 (来自 map.json)
    q_coord = generate_coordinate_queries(zones)
    q_access = generate_access_queries(zones, default_access)
    q_point = generate_point_queries(zones)
    q_hierarchy = generate_hierarchy_queries(zones)
    q_height = generate_height_queries(zones)
    
    # 规则知识 (来自 map_rules_cn.json)
    q_rules = generate_rule_queries(rules_data)
    
    # 拓扑知识 (来自 map_accessible_adjcent.json)
    q_topology = generate_topology_queries(adj_list, r2_accessible)
    
    # 推理知识 (综合)
    q_pathfinding = generate_pathfinding_queries(adj_list, r2_accessible, zones_map)
    
    all_datasets.extend(q_coord)
    all_datasets.extend(q_access)
    all_datasets.extend(q_point)
    all_datasets.extend(q_hierarchy)
    all_datasets.extend(q_height)
    all_datasets.extend(q_rules)
    all_datasets.extend(q_topology)
    all_datasets.extend(q_pathfinding)
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    output_path_jsonl = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME_JSONL)
    output_path_json = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME_JSON)

    # 保存为JSONL格式(适合LoRA训练)
    print(f"正在写入 {output_path_jsonl} ...")
    with open(output_path_jsonl, 'w', encoding='utf-8') as f:
        for item in all_datasets:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    # 也保存为JSON格式(方便查看)
    print(f"正在写入 {output_path_json} ...")
    with open(output_path_json, 'w', encoding='utf-8') as f:
        json.dump(all_datasets, f, ensure_ascii=False, indent=2)
        
    print(f"\n数据集生成完成!")
    print(f"总共生成 {len(all_datasets)} 条训练数据")
    print(f"输出文件:")
    print(f"  - {output_path_jsonl} (用于训练)")
    print(f"  - {output_path_json} (用于查看)")
        
    # 统计各类数据数量
    print(f"\n数据分布:")
    print(f"  - 坐标查询: {len(q_coord)} 条")
    print(f"  - 访问权限: {len(q_access)} 条")
    print(f"  - 坐标归属: {len(q_point)} 条")
    print(f"  - 层级关系: {len(q_hierarchy)} 条")
    print(f"  - 高度信息: {len(q_height)} 条")
    print(f"  - 地图规则: {len(q_rules)} 条")
    print(f"  - 拓扑邻接: {len(q_topology)} 条")
    print(f"  - 路径推理: {len(q_pathfinding)} 条")

if __name__ == "__main__":
    main()

