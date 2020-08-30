use cgmath::Deg;
use cgmath::Euler;
use cgmath::InnerSpace;
use cgmath::Quaternion;
use cgmath::Vector3;

fn new_vec(x: f32, y: f32, z: f32) -> Vector3<f32> {
    Vector3 { x, y, z }
}

fn main() {
    let tree = VoxelTree::allocate_VoxelTree(120000);

    let ad = tree.allocate_tree_node();
    tree.ref_node(ad).material = 1;
    tree.ref_node(ad).right = 7;
    tree.ref_node(ad).group = 53;

    println!("{:?}", tree.get_node(ad));

    }/*
    tree.put_node_address1(ad, 1234);
    tree.put_node_address2(ad, 4321);

    println!("address1 : {}", tree.get_node_address1(ad));
    println!("address2 : {}", tree.get_node_address2(ad));

    tree.put_leaf_bounds(ad, [5.0; 6]);
    println!("bounds {:?}", tree.get_leaf_bounds(ad));

    let mut data = vec![
        abox(-1.2, 2.0, 0.0),
        abox(3.0, 0.0, 0.0),
        abox(-2.0, -4.0, -1.5),
        abox(-2.0, -1.0, 6.2),
        abox(-4.0, -1.0, 0.7),
        //abox(-4.0, 4.0, -1.4),
        //abox(-7.0, 2.0, 7.9),
    ];

    fn do_x(
        data: &mut Vec<AABB>,
        mut level: usize,
        tree: &KDTree,
        node: u32,
        mut last_split: Vector3<f32>,
        rope: [u32; 6],
        bounds: [f32; 6],
    ) {
        if data.len() <= 0 {
            do_z(data, 7, tree, node, last_split, rope, bounds);
            return;
        }
        if data.len() == 1 {
            level += 1;
        }
        let (mut data2, median) = split_x(data, &mut last_split);
        tree.put_node_head(node, 1);
        tree.put_node_divider(node, median);

        let ad = tree.allocate_tree_node();
        let ad2 = tree.allocate_tree_node();

        tree.put_node_address1(node, ad);
        tree.put_node_address2(node, ad2);

        let mut rope1 = rope;
        let mut rope2 = rope;

        rope1[1] = ad2;
        rope2[0] = ad;

        let mut bounds1 = bounds;
        let mut bounds2 = bounds;
        bounds1[1] = median.max(bounds1[1]);
        bounds2[0] = median.min(bounds2[0]);

        do_y(data, level, tree, ad, last_split, rope1, bounds1);
        do_y(&mut data2, level, tree, ad2, last_split, rope2, bounds2);
    }

    fn do_y(
        data: &mut Vec<AABB>,
        mut level: usize,
        tree: &KDTree,
        node: u32,
        mut last_split: Vector3<f32>,
        rope: [u32; 6],
        bounds: [f32; 6],
    ) {
        if data.len() <= 0 {
            do_z(data, 7, tree, node, last_split, rope, bounds);
            return;
        }
        if data.len() == 1 {
            level += 1;
        }
        let (mut data2, median) = split_y(data, &mut last_split);
        tree.put_node_head(node, 2);
        tree.put_node_divider(node, median);

        let ad = tree.allocate_tree_node();
        let ad2 = tree.allocate_tree_node();

        tree.put_node_address1(node, ad);
        tree.put_node_address2(node, ad2);

        let mut rope1 = rope;
        let mut rope2 = rope;

        rope1[3] = ad2;
        rope2[2] = ad;

        let mut bounds1 = bounds;
        let mut bounds2 = bounds;
        bounds1[3] = median.max(bounds1[3]);
        bounds2[2] = median.min(bounds2[2]);

        do_z(data, level, tree, ad, last_split, rope1, bounds1);
        do_z(&mut data2, level, tree, ad2, last_split, rope2, bounds2);
    }

    fn do_z(
        data: &mut Vec<AABB>,
        mut level: usize,
        tree: &KDTree,
        node: u32,
        mut last_split: Vector3<f32>,
        rope: [u32; 6],
        bounds: [f32; 6],
    ) {
        if level > 6 || data.len() <= 0 {
            if level <= 7 {
                tree.put_node_head(node, 3);
                tree.put_node_divider(node, 0.0);
                let ad = tree.allocate_tree_leaf();
                tree.put_node_address1(node, ad);
                tree.put_node_address2(node, ad);
                do_z(data, level + 1, tree, ad, last_split, rope, bounds);
                return;
            }
            tree.put_node_head(node, 0);
            tree.put_leaf_rope(node, rope);
            tree.put_leaf_bounds(node, bounds);
            if data.len() > 0 {
                tree.put_node_material(node, 1);
            } else {
                tree.put_node_material(node, 0);
            }
            return;
        }

        let (mut data2, median) = split_z(data, &mut last_split);

        tree.put_node_head(node, 3);
        tree.put_node_divider(node, median);

        let ad = tree.allocate_tree_node();
        let ad2 = tree.allocate_tree_node();

        tree.put_node_address1(node, ad);
        tree.put_node_address2(node, ad2);

        let mut rope1 = rope;
        let mut rope2 = rope;

        rope1[5] = ad2;
        rope2[4] = ad;

        let mut bounds1 = bounds;
        let mut bounds2 = bounds;
        bounds1[5] = median.max(bounds1[5]);
        bounds2[4] = median.min(bounds2[4]);

        do_x(data, level, tree, ad, last_split, rope1, bounds1);
        do_x(&mut data2, level, tree, ad2, last_split, rope2, bounds2);
    }

    for x in data.iter() {
        println!("{:?}", x);
    }
    println!("");

    let instant = std::time::Instant::now();
    do_x(
        &mut data.clone(),
        0,
        &tree,
        ad,
        new_vec(-f32::MAX, -f32::MAX, -f32::MAX),
        [0u32; 6],
        [
            f32::MAX,
            -f32::MAX,
            f32::MAX,
            -f32::MAX,
            f32::MAX,
            -f32::MAX,
        ],
    );
    println!("Build time: {} ns", instant.elapsed().as_nanos());

    let width: u32 = 256;
    let height: u32 = 256;
    let depth: u32 = 1;

    let campos = new_vec(0.0, 0.0, 0.0);
    let camrot = new_vec(0.0, 0.0, 0.0);

    println!(
        "{}",
        traverse(&tree, ad, new_vec(1.9, 0.0, 0.0), new_vec(1.0, 0.0, 0.0))
    );

    //return;

    use std::sync::Mutex;
    let count_mutex = Mutex::new(0u32);

    use rayon::iter::IntoParallelIterator;
    use rayon::iter::ParallelIterator;
    (0..depth).into_par_iter().for_each(move |z| {
        let mut buffer: Vec<u8> = vec![0; (width * height * 4) as usize]; // Generate the image data;
        let instant = std::time::Instant::now();
        for x in 0..width {
            for y in 0..height {
                let fx = x as f32 / width as f32;
                let fy = y as f32 / height as f32;
                let fz = z as f32 / depth as f32;
                let rotation = Quaternion::from(Euler {
                    x: Deg(camrot.x),
                    y: Deg(camrot.y),
                    z: Deg(camrot.z),
                });

                let pos = rotation * new_vec(fx * 20.0 - 10.0, fy * 20.0 - 10.0, fz * 10.0 - 10.0);
                println! ("new///////////////////////////////////////////////////////");
                if traverse(
                    &tree,
                    ad,
                    new_vec(pos.x, pos.y, -10.0),
                    new_vec(0.0, 0.0, 1.0),
                ) {
                    let index = ((y * width + x) * 4 as u32) as usize;
                    let colour: [u8; 3] = [200, 50, 25];

                    buffer[index] = colour[0];
                    buffer[index + 1] = colour[1];
                    buffer[index + 2] = colour[2];
                    buffer[index + 3] = 255;
                }
            }
        }
        let avg_tra_time = instant.elapsed().as_nanos() / (width * height) as u128;
        // Save the buffer as "image.png"
        image::ImageBuffer::<image::Rgba<u8>, _>::from_raw(width, height, &buffer[..])
            .unwrap()
            .save(format!("image/{}.png", z))
            .unwrap();
        let mut count = count_mutex.lock().unwrap();
        println!(
            "{}/{}    Avg traverse time: {} ns",
            *count + 1,
            depth,
            avg_tra_time
        );
        *count += 1;
    });
}

fn traverse(tree: &KDTree, start: u32, mut vec: Vector3<f32>, dir: Vector3<f32>) -> bool {
    let mut addr = start;
    loop {
        let head = tree.get_node_head(addr);

        println!("addr {}, head {}, vec {:?}, dir {:?}", addr, head, vec, dir);
        if vec.magnitude2().is_nan() {
            panic!();
        }

        if head == 0 {
            if tree.get_node_material(addr) != 0 {
                return true;
            }

            let rope = tree.get_leaf_rope(addr);
            let bounds = tree.get_leaf_bounds(addr);
            println!("{:?}", rope);
            println!("{:?}", bounds);
            let mut txp = (bounds[0] - vec.x) / dir.x;
            let mut txn = (bounds[1] - vec.x) / dir.x;

            let mut typ = (bounds[2] - vec.y) / dir.y;
            let mut tyn = (bounds[3] - vec.y) / dir.y;

            let mut tzp = (bounds[4] - vec.z) / dir.z;
            let mut tzn = (bounds[5] - vec.z) / dir.z;

            println!("{}", txp);
            println!("{}", txn);
            println!("{}", typ);
            println!("{}", tyn);
            println!("{}", tzp);
            println!("{}", tzn);

            println!("stuff");

            txp += 1.0;
            txn += 1.0;

            typ += 1.0;
            tyn += 1.0;

            tzp += 1.0;
            tzn += 1.0;

            txp *= (txp > 0.0) as u8 as f32;
            txn *= (txn > 0.0) as u8 as f32;

            typ *= (typ > 0.0) as u8 as f32;
            tyn *= (tyn > 0.0) as u8 as f32;

            tzp *= (tzp > 0.0) as u8 as f32;
            tzn *= (tzn > 0.0) as u8 as f32;

            txp -= 1.0;
            txn -= 1.0;

            typ -= 1.0;
            tyn -= 1.0;

            tzp -= 1.0;
            tzn -= 1.0;

            txp += (txp < 0.0) as u8 as f32 * f32::MAX * 2.0;
            txn += (txn < 0.0) as u8 as f32 * f32::MAX * 2.0;

            typ += (typ < 0.0) as u8 as f32 * f32::MAX * 2.0;
            tyn += (tyn < 0.0) as u8 as f32 * f32::MAX * 2.0;

            tzp += (tzp < 0.0) as u8 as f32 * f32::MAX * 2.0;
            tzn += (tzn < 0.0) as u8 as f32 * f32::MAX * 2.0;

            let closest = txp.min(txn.min(typ.min(tyn.min(tzp.min(tzn)))));

            println!("{}", txp);
            println!("{}", txn);
            println!("{}", typ);
            println!("{}", tyn);
            println!("{}", tzp);
            println!("{}", tzn);

            println!("closest {}", closest);

            let mut index = 0;
            if txp == closest {
                index = 0;
            } else if txn == closest {
                index = 1;
            } else if typ == closest {
                index = 2;
            } else if tyn == closest {
                index = 3;
            } else if tzp == closest {
                index = 4;
            } else if tzn == closest {
                index = 5;
            }

            const a_small_nudge: f32 = 0.002;

            addr = rope[index];
            vec += dir * (closest.abs() + a_small_nudge);

            println!("new address is {}!", addr);

            if closest.is_infinite() {
                addr = 0;
            }

            if addr == 0 {
                return false;
            }

            continue;
        }
        let divider = tree.get_node_divider(addr);
        if divider.is_nan() {
            panic!();
        }
        if (head == 1 && vec.x > divider)
            || (head == 2 && vec.y > divider)
            || (head == 3 && vec.z > divider)
        {
            addr = tree.get_node_address1(addr);
        }
        if (head == 1 && vec.x <= divider)
            || (head == 2 && vec.y <= divider)
            || (head == 3 && vec.z <= divider)
        {
            addr = tree.get_node_address2(addr);
        }
    }
    return false;
}

#[derive(Clone, Copy, Debug)]
struct AABB {
    min: Vector3<f32>,
    max: Vector3<f32>,
}

fn abox(x: f32, y: f32, z: f32) -> AABB {
    AABB {
        min: Vector3 {
            x: x - 1.0,
            y: y - 1.0,
            z: z - 1.0,
        },
        max: Vector3 {
            x: x + 1.0,
            y: y + 1.0,
            z: z + 1.0,
        },
    }
}

fn split_x(aabbs: &mut Vec<AABB>, last: &mut Vector3<f32>) -> (Vec<AABB>, f32) {
    if aabbs.len() <= 0 {
        return (Vec::new(), 0.0);
    }
    let mut it = Vec::new();
    for a in aabbs.iter() {
        it.push(a.max.x);
        if a.min.x != last.x {
            it.push(a.min.x);
        }
    }
    let median = fast_median(it);
    last.x = median;

    let mut vec = Vec::new();
    vec.reserve(aabbs.len() / 2);
    let mut index = 0;
    for i in 0..aabbs.len() {
        if aabbs[i].max.x > median {
            aabbs[index] = aabbs[i];
            index += 1;
        }
        if aabbs[i].min.x < median {
            vec.push(aabbs[i]);
        }
    }
    aabbs.resize(index, abox(0.0, 0.0, 0.0));
    (vec, median)
}

fn split_y(aabbs: &mut Vec<AABB>, last: &mut Vector3<f32>) -> (Vec<AABB>, f32) {
    if aabbs.len() <= 0 {
        return (Vec::new(), 0.0);
    }
    let mut it = Vec::new();
    for a in aabbs.iter() {
        it.push(a.max.y);
        if a.min.y != last.y {
            it.push(a.min.y);
        }
    }
    let median = fast_median(it);
    last.y = median;

    let mut vec = Vec::new();
    vec.reserve(aabbs.len() / 2);
    let mut index = 0;
    for i in 0..aabbs.len() {
        if aabbs[i].max.y > median {
            aabbs[index] = aabbs[i];
            index += 1;
        }
        if aabbs[i].min.y < median {
            vec.push(aabbs[i]);
        }
    }
    aabbs.resize(index, abox(0.0, 0.0, 0.0));
    (vec, median)
}

fn split_z(aabbs: &mut Vec<AABB>, last: &mut Vector3<f32>) -> (Vec<AABB>, f32) {
    if aabbs.len() <= 0 {
        return (Vec::new(), 0.0);
    }

    let mut it = Vec::new();
    for a in aabbs.iter() {
        it.push(a.max.z);
        if a.min.z != last.z {
            it.push(a.min.z);
        }
    }
    let median = fast_median(it);
    last.z = median;

    let mut vec = Vec::new();
    vec.reserve(aabbs.len() / 2);
    let mut index = 0;
    for i in 0..aabbs.len() {
        if aabbs[i].max.z > median {
            aabbs[index] = aabbs[i];
            index += 1;
        }
        if aabbs[i].min.z < median {
            vec.push(aabbs[i]);
        }
    }
    aabbs.resize(index, abox(0.0, 0.0, 0.0));
    (vec, median)
}

fn fast_median(data: Vec<f32>) -> f32 {
    fn calculate_mean_and_stddv<'a, I>(iterator: I) -> (f32, f32)
    where
        I: Iterator<Item = &'a f32>,
    {
        fn update_variance(
            mut count: usize,
            mut mean: f32,
            mut m2: f32,
            new_value: f32,
        ) -> (usize, f32, f32) {
            count += 1;
            let delta = new_value - mean;
            mean += delta / (count as f32);
            let delta2 = new_value - mean;
            m2 += delta * delta2;

            (count, mean, m2)
        }
        fn finalize_variance(mut count: usize, mut mean: f32, mut m2: f32) -> (f32, f32) {
            // (mean, variance)
            if count < 2 {
                return (mean, 0.0);
            } else {
                return (mean, m2 / (count as f32));
            }
        }

        let mut count = 0;
        let mut mean = 0.0;
        let mut m2 = 0.0;
        for x in iterator {
            let (r1, r2, r3) = update_variance(count, mean, m2, *x);
            count = r1;
            mean = r2;
            m2 = r3;
        }
        let (mean, variance) = finalize_variance(count, mean, m2);
        (mean, variance.sqrt())
    }

    let (mean, stddv) = calculate_mean_and_stddv(data.iter());

    if data.len() <= 0 {
        return 0.0;
    }
    if data.len() == 2 {
        if data[0] < data[1] {
            return data[0];
        } else {
            return data[1];
        }
    }

    if data.len() <= 4 {
        let mut total = 0.0;
        for x in data.iter() {
            total += x;
        }
        return total / (data.len() as f32);
    }

    let data_len = data.len();
    debug_assert!(data_len > 4);

    let def = data[0];

    let bin_count: usize = data_len;

    let mut bottomcount = 0;
    let mut bincounts = vec![0; bin_count + 1];

    let scalefactor = bin_count as f32 / (2.0 * stddv);
    let leftend = mean - stddv;
    let rightend = mean + stddv;

    for d in data.iter() {
        if *d < leftend {
            bottomcount += 1;
        } else if *d < rightend {
            let bin = ((d - leftend) * scalefactor) as usize;
            bincounts[bin] += 1;
        }
    }

    if data_len & 1 != 0 {
        //odd
        let k = (data_len + 1) / 2;
        let mut count = bottomcount;

        for i in 0..(bin_count + 1) {
            count += bincounts[i];

            if count >= k {
                return (i as f32 + 0.5) / scalefactor + leftend;
            }
        }
    } else {
        let k = data_len / 2;
        let mut count = bottomcount;

        for i in 0..(bin_count + 1) {
            count += bincounts[i];
            if count >= k {
                let mut j = i;
                while count == k {
                    j += 1;
                    count += bincounts[j];
                }
                return (i + j + 1) as f32 / (2.0 * scalefactor) + leftend;
            }
        }
    }
    def
}
 */
#[derive(Debug, Clone, Copy)]
struct VoxelNode {
    group: u16,
    material: u16,
    right: u16,
    left: u16,
    up: u16,
    down: u16,
    forward: u16,
    backward: u16,
}

#[derive(Clone, Copy)]
struct VoxelTree {
    base: *mut u16,
    allocator: *mut u32,
    size: *mut u32,
    layout: std::alloc::Layout,
}

unsafe impl Send for VoxelTree {}
unsafe impl Sync for VoxelTree {}

impl VoxelTree {
    pub fn allocate_VoxelTree(size_in_2_byte_chunks: u32) -> VoxelTree {
        unsafe {
            let layout =
                std::alloc::Layout::from_size_align(4 + size_in_2_byte_chunks as usize,
                    std::mem::size_of::<VoxelNode>()).unwrap();

            let base: *mut u16 = std::alloc::alloc(layout) as *mut u16;

            let allocator = base as *mut u32;
            *allocator = 2;

            let size = base.offset(*allocator as isize) as *mut u32;
            *allocator += 2;
            *size = 4 + size_in_2_byte_chunks;

            VoxelTree {
                base,
                allocator,
                size,
                layout,
            }
        }
    }

    pub fn allocate_tree_node(&self) -> u32 {
        unsafe {
            let addr = *self.allocator;
            *self.allocator += (std::mem::size_of::<VoxelNode>() / 2) as u32;
            debug_assert!(
                *self.allocator < *self.size,
                format!(
                    "KDTree ran out of memory. {} > {}",
                    *self.allocator, *self.size
                )
            );
            *self.base.offset(addr as isize) = 0;
            addr
        }
    }
    
    pub fn ref_node(&self, address: u32) -> &mut VoxelNode {
        unsafe {
            (self.base.offset(address as isize) as *mut VoxelNode).as_mut().unwrap()
        }
    }
    
    pub fn put_node(&self, address: u32, value: VoxelNode) {
        unsafe {
            let ptr = (self.base.offset(address as isize) as *mut VoxelNode);
            *(ptr as *mut VoxelNode) = value;
        }
    }

    pub fn get_node(&self, address: u32) -> VoxelNode {
        unsafe {
            let ptr = (self.base.offset(address as isize) as *mut VoxelNode);
            *(ptr as *mut VoxelNode)
        }
    }
    
    #[inline(always)]
    pub fn get_sofar_size(&self) -> u32 {
        unsafe { *self.allocator }
    }
    #[inline(always)]
    pub fn get_allocation_size(&self) -> u32 {
        unsafe { *self.size }
    }
}

