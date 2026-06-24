import cv2
import numpy as np

class VisionProcessor:
    def __init__(self, canvas_center_x=0.45, canvas_center_y=0.0, draw_width=0.15, draw_height=0.20):
        self.center_x = canvas_center_x
        self.center_y = canvas_center_y
        self.width = draw_width
        self.height = draw_height

    def extract_drawing_paths(self, image_path, min_path_length=5):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Can not read {image_path}")

        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        paths = []
        h, w = img.shape

        # 1. Tìm tỷ lệ thu nhỏ tối đa cho phép trên cả 2 chiều
        scale_x = self.width / w
        scale_y = self.height / h
        
        # Lấy tỷ lệ nhỏ hơn để đảm bảo ảnh không bị tràn ra ngoài giấy
        scale = min(scale_x, scale_y) 

        # 2. Kích thước bức tranh thực tế trên bàn MuJoCo (mét)
        actual_draw_width = w * scale
        actual_draw_height = h * scale

        # 3. Tính mốc tọa độ trên cùng bên trái để bức tranh nằm chính giữa tờ giấy
        start_x = self.center_x - (actual_draw_width / 2)
        start_y = self.center_y + (actual_draw_height / 2) 
        # =========================================================

        for contour in contours:
            if len(contour) < min_path_length:
                continue

            path = []
            for point in contour:
                u, v = point[0] 
                
                # Ánh xạ chuẩn xác: Tọa độ gốc + (pixel * tỷ lệ mét/pixel)
                x_metric = start_x + (u * scale)
                y_metric = start_y - (v * scale)
                
                path.append([x_metric, y_metric])
            
            paths.append(np.array(path))

        return paths, edges

    def show_preview(self, edges_image):
        # Đảo ngược màu: Nền trắng, nét đen để dễ nhìn giống giấy thật
        preview_img = cv2.bitwise_not(edges_image)
        cv2.imshow("Robot Drawing Preview - Press 'q' or Esc to exit", preview_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    test_image = "data/images/pikachu.jpg" 
    
    try:
        processor = VisionProcessor()
        paths, edge_map = processor.extract_drawing_paths(test_image)
        
        print(f"✅ Done processing image! Found {len(paths)} drawing paths.")
        print(f"First path has {len(paths[0])} points.")
        print(f"First point: X={paths[0][0][0]:.4f}m, Y={paths[0][0][1]:.4f}m")
        
        processor.show_preview(edge_map)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
