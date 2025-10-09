import json
import cv2
from os.path import join as pjoin
from detect_merge.Element import Element

def show_elements(org_img, eles, show=False, win_name='element', wait_key=0, shown_resize=None, line=2):
    color_map = {'Text':(0, 0, 255), 'Compo':(0, 255, 0), 'Block':(0, 255, 0), 'Text Content':(255, 0, 255)}
    img = org_img.copy()
    for ele in eles:
        color = color_map[ele.category]
        ele.visualize_element(img, color, line)
    img_resize = img
    if shown_resize is not None:
        img_resize = cv2.resize(img, shown_resize)
    if show:
        # cv2.imshow(win_name, img_resize)
        # cv2.waitKey(wait_key)
        # if wait_key == 0:
        #     cv2.destroyWindow(win_name)
        pass
    return img_resize

def reassign_ids(elements):
    for i, element in enumerate(elements):
        element.id = i

def refine_elements(compos, texts, intersection_bias=(2, 2), containment_ratio=0.8):
    elements = []
    contained_texts = set()  
    used_compos = set()   

    for compo in compos:
        local_contained_texts = []
        to_delete = False

        for text in texts:
            inter, _, ioa, iob = compo.calc_intersection_area(text, bias=intersection_bias)

            # Case 1: text completely contains the component → remove component
            if iob >= 1.0:
                to_delete = True
                break

            # Case 2: component contains text OR both have high overlap ratio → merge
            if ioa >= containment_ratio and iob >= containment_ratio:
                local_contained_texts.append(text)
                contained_texts.add(text)

            # Case 3: component partially overlaps and covers text (but not Block category)
            elif iob >= containment_ratio and compo.category != 'Block':
                local_contained_texts.append(text)
                contained_texts.add(text)

        if not to_delete:
            # Attach contained texts as children of the component
            for t in local_contained_texts:
                t.parent_id = compo.id
            compo.children += local_contained_texts
            elements.append(compo)
            used_compos.add(compo)

    # Add remaining components that were not used
    for compo in compos:
        if compo not in used_compos:
            elements.append(compo)

    # Add texts that were not contained in any component
    for text in texts:
        if text not in contained_texts:
            elements.append(text)

    return elements

def check_containment(elements):
    for i in range(len(elements) - 1):
        for j in range(i + 1, len(elements)):
            relation = elements[i].element_relation(elements[j], bias=(2, 2))
            if relation == -1:
                elements[j].children.append(elements[i])
                elements[i].parent_id = elements[j].id
            if relation == 1:
                elements[i].children.append(elements[j])
                elements[j].parent_id = elements[i].id


# Original version
# def connect_minimal_tree_edges_with_children(elements, canvas):
#     connections = [] 

#     def get_edge_points(ele):
#         left = (ele.col_min, (ele.row_min + ele.row_max) // 2)
#         right = (ele.col_max, (ele.row_min + ele.row_max) // 2)
#         top = ((ele.col_min + ele.col_max) // 2, ele.row_min)
#         bottom = ((ele.col_min + ele.col_max) // 2, ele.row_max)
#         return [left, right, top, bottom]

#     def distance(p1, p2):
#         return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

#     id2ele = {e.id: e for e in elements}
#     child_id_set = set()
#     child_groups = []

#     for ele in elements:
#         if hasattr(ele, "children") and ele.children:
#             group = []
#             for child in ele.children:
#                 child_id_set.add(child.id)
#                 group.append(child)
#             child_groups.append(group)

#     main_elements = [e for e in elements if e.id not in child_id_set]

#     def connect_group(group_elements):
#         """
#         Connect elements in a group using a minimal spanning tree
#         based on shortest distances between edge points.
#         """
#         edge_points = [get_edge_points(e) for e in group_elements]
#         n = len(group_elements)
#         edges = []

#         for i in range(n):
#             for j in range(i + 1, n):
#                 min_d = float('inf')
#                 best_pair = None
#                 for p1 in edge_points[i]:
#                     for p2 in edge_points[j]:
#                         d = distance(p1, p2)
#                         if d < min_d:
#                             min_d = d
#                             best_pair = (p1, p2)
#                 edges.append((min_d, i, j, best_pair))
#         edges.sort()

#         # Disjoint-set union-find for MST
#         parent = list(range(n))

#         def find(x):
#             while parent[x] != x:
#                 parent[x] = parent[parent[x]]
#                 x = parent[x]
#             return x

#         def union(x, y):
#             rx, ry = find(x), find(y)
#             if rx != ry:
#                 parent[rx] = ry
#                 return True
#             return False

#         # Kruskal’s algorithm: add edges if they connect different components
#         for _, i, j, (pt1, pt2) in edges:
#             if union(i, j):
#                 cv2.line(canvas, pt1, pt2, (100, 200, 255), 1)
#                 connections.append({
#                     "from": [int(pt1[0]), int(pt1[1])],
#                     "to": [int(pt2[0]), int(pt2[1])]
#                 })

#     if len(main_elements) >= 2:
#         connect_group(main_elements)

#     for group in child_groups:
#         if len(group) >= 2:
#             connect_group(group)

#     return connections


def connect_minimal_tree_edges_with_children(elements, canvas):
    """
    Connects elements and their children using a minimal spanning tree
    with improved shortest-link calculation (handles horizontal/vertical
    separations and overlapping cases).
    
    Args:
        elements: list of element objects, each with bounding box attributes 
                  (row_min, row_max, col_min, col_max, id, children).
        canvas:   OpenCV image canvas where the connections will be drawn.
    
    Returns:
        connections: list of dictionaries storing connected line endpoints.
    """
    import math
    import cv2

    connections = []

    # Build element lookup and collect child groups
    id2ele = {e.id: e for e in elements}
    child_id_set = set()
    child_groups = []

    for ele in elements:
        if hasattr(ele, "children") and ele.children:
            group = []
            for child in ele.children:
                child_id_set.add(child.id)
                group.append(child)
            child_groups.append(group)

    # Identify main elements (not in any child set)
    main_elements = [e for e in elements if e.id not in child_id_set]

    def shortest_link(e1, e2):
        """
        Compute the shortest link between two elements based on three cases:
        
        (1) Horizontal separation + vertical overlap  
            → Connect the midpoint of the shorter element's vertical edge 
              to the opposite element's vertical edge at the same height.
        
        (2) Vertical separation + horizontal overlap  
            → Connect the midpoint of the shorter element's horizontal edge 
              to the opposite element's horizontal edge at the same width.
        
        (3) Separated in both directions  
            → Connect the closest pair of corner points.
        """
        x1_min, y1_min, x1_max, y1_max = e1.col_min, e1.row_min, e1.col_max, e1.row_max
        x2_min, y2_min, x2_max, y2_max = e2.col_min, e2.row_min, e2.col_max, e2.row_max

        w1, h1 = x1_max - x1_min, y1_max - y1_min
        w2, h2 = x2_max - x2_min, y2_max - y2_min

        # Case 1: Horizontal separation with vertical overlap
        if x1_max < x2_min and not (y1_max < y2_min or y2_max < y1_min):
            mid_y = (y1_min + y1_max) // 2 if h1 <= h2 else (y2_min + y2_max) // 2
            return (x1_max, mid_y), (x2_min, mid_y)

        if x2_max < x1_min and not (y1_max < y2_min or y2_max < y1_min):
            mid_y = (y1_min + y1_max) // 2 if h1 <= h2 else (y2_min + y2_max) // 2
            return (x2_max, mid_y), (x1_min, mid_y)

        # Case 2: Vertical separation with horizontal overlap
        if y1_max < y2_min and not (x1_max < x2_min or x2_max < x1_min):
            mid_x = (x1_min + x1_max) // 2 if w1 <= w2 else (x2_min + x2_max) // 2
            return (mid_x, y1_max), (mid_x, y2_min)

        if y2_max < y1_min and not (x1_max < x2_min or x2_max < x1_min):
            mid_x = (x1_min + x1_max) // 2 if w1 <= w2 else (x2_min + x2_max) // 2
            return (mid_x, y2_max), (mid_x, y1_min)

        # Case 3: Separated in both directions → use closest corner points
        corners1 = [(x1_min, y1_min), (x1_min, y1_max),
                    (x1_max, y1_min), (x1_max, y1_max)]
        corners2 = [(x2_min, y2_min), (x2_min, y2_max),
                    (x2_max, y2_min), (x2_max, y2_max)]

        min_d, best_pair = float('inf'), None
        for p1 in corners1:
            for p2 in corners2:
                d = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
                if d < min_d:
                    min_d, best_pair = d, (p1, p2)
        return best_pair

    def connect_group(group_elements):
        """
        Build a minimal spanning tree for a group of elements
        using Kruskal’s algorithm with shortest-link distances.
        """
        n = len(group_elements)
        edges = []

        # Compute all candidate edges with distances
        for i in range(n):
            for j in range(i + 1, n):
                pt1, pt2 = shortest_link(group_elements[i], group_elements[j])
                d = math.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])
                edges.append((d, i, j, pt1, pt2))
        edges.sort()

        # Union-Find (Disjoint Set Union)
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[rx] = ry
                return True
            return False

        # Apply Kruskal’s MST
        for _, i, j, pt1, pt2 in edges:
            if union(i, j):
                cv2.line(canvas, pt1, pt2, (100, 200, 255), 1)
                connections.append({
                    "from": [int(pt1[0]), int(pt1[1])],
                    "to": [int(pt2[0]), int(pt2[1])]
                })

    # Connect main elements
    if len(main_elements) >= 2:
        connect_group(main_elements)

    # Connect each child group independently
    for group in child_groups:
        if len(group) >= 2:
            connect_group(group)

    return connections

def remove_containers(compos):
    to_remove = set()
    for i, outer in enumerate(compos):
        for j, inner in enumerate(compos):
            if i != j:
                relation = outer.element_relation(inner)
                if relation == 1:  
                    to_remove.add(i)
    return [c for idx, c in enumerate(compos) if idx not in to_remove]

def remove_compos_inside_texts(compos, texts, area_threshold=0.8):
    to_remove = set()
    for i, compo in enumerate(compos):
        for text in texts:
            relation = compo.element_relation(text)
            if relation == -1: 
                to_remove.add(i)
                break
            else:
                inter, _, ioa, _ = compo.calc_intersection_area(text)
                if ioa >= area_threshold:  
                    to_remove.add(i)
                    break
    return [c for idx, c in enumerate(compos) if idx not in to_remove]

def save_elements(output_file, elements, connections=None):
    components = {
        'compos': [],
        'texts': [],
        'connections': connections if connections else []
    }

    for ele in elements:
        c = ele.wrap_info()
        if c['class'] == 'Text':
            components['texts'].append(c)
        else:
            components['compos'].append(c)

    json.dump(components, open(output_file, 'w'), indent=4, ensure_ascii=False)
    return components



def merge_EfficientUI(img_path, compo_path, text_path, merge_root=None, show=False, wait_key=0):
    compo_json = json.load(open(compo_path, 'r'))
    text_json = json.load(open(text_path, 'r'))

    ele_id = 0
    compos = []
    for compo in compo_json['compos']:
        element = Element(ele_id, (compo['column_min'], compo['row_min'], compo['column_max'], compo['row_max']), compo['class'])
        compos.append(element)
        ele_id += 1
    compos = remove_containers(compos)
    
    texts = []
    for text in text_json['texts']:
        element = Element(ele_id, (text['column_min'], text['row_min'], text['column_max'], text['row_max']), 'Text', text_content=text['content'])
        texts.append(element)
        ele_id += 1

    img = cv2.imread(img_path)
    img_resize = cv2.resize(img, (compo_json['img_shape'][1], compo_json['img_shape'][0]))

    if compo_json['img_shape'] != text_json['img_shape']:
        print('[Warning] The image shapes of components and texts are different, resizing texts to match components.')
        resize_ratio = compo_json['img_shape'][0] / text_json['img_shape'][0]
        for text in texts:
            text.resize(resize_ratio)

    compos = remove_compos_inside_texts(compos, texts, area_threshold=0.8)

    elements = refine_elements(compos, texts, intersection_bias=(2, 2), containment_ratio=0.8)

    reassign_ids(elements)
    check_containment(elements)
    board = show_elements(img_resize, elements, show=show, win_name='elements after merging', wait_key=wait_key)

    connections = connect_minimal_tree_edges_with_children(elements, board)

    name = img_path.replace('\\', '/').split('/')[-1][:-4]
    components = save_elements(pjoin(merge_root, name + '.json'), elements, connections=connections)
    cv2.imwrite(pjoin(merge_root, name + '.jpg'), board)
    print('[Merge Completed] Input: %s Output: %s' % (img_path, pjoin(merge_root, name + '.jpg')))
    return board, components
