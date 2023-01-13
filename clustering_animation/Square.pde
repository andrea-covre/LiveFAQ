class Corner {
  float x;
  float y;
}

class Square {
  Corner tl;
  Corner tr;
  Corner bl;
  Corner br;
  color bg_color;
  int w = int((width - 2 * border_spacing - ((squares.length - 1) * intra_spacing)) / squares.length);
  int h = int((height - 2 * border_spacing - ((squares[0].length - 1) * intra_spacing)) / squares[0].length);
  int cluster_id;
  boolean visible = true;
  String text;
  
  boolean is_center_init;
  float center_x;
  float center_y;
  
  Square() {
    this.tl = new Corner();
    this.tr = new Corner();
    this.bl = new Corner();
    this.br = new Corner();
    this.is_center_init = false;
    this.center_x = 0;
    this.center_y = 0;
    this.text = "";
  }
  
  void display() {
    fill(this.bg_color, 120);
    rect(this.tl.x, this.tl.y, this.br.x, this.br.y, corner_radius);
  }
  
  float[] get_center() {
    if (this.is_center_init == false) {
      this.center_x = (this.br.x - this.tl.x) / 2 + this.tl.x;
      this.center_y = (this.br.y - this.tl.y) / 2 + this.tl.y;
      this.is_center_init = true;
    }
    float[] center = {this.center_x, this.center_y};
    
    return center;
  }
}

void init_squares() {

  for (int col = 0; col < squares.length; col++) {
    for (int row = 0; row < squares[0].length; row++) {
      Square square = new Square();
      
      square.bg_color = bg_squares[row + col * squares[0].length];
      
      // TL corner
      square.tl.x = col * (square.w + intra_spacing) + border_spacing;
      square.tl.y = row * (square.h + intra_spacing) + border_spacing;
      
      // TR corner
      square.tr.x = col * (square.w + intra_spacing) + square.w + border_spacing;
      square.tr.y = row * (square.h + intra_spacing) + border_spacing;
      
      // BL corner
      square.bl.x = col * (square.w + intra_spacing) + border_spacing;
      square.bl.y = row * (square.h + intra_spacing) + square.h + border_spacing;
      
      // BR corner
      square.br.x = col * (square.w + intra_spacing) + square.w + border_spacing;
      square.br.y = row * (square.h + intra_spacing) + square.h + border_spacing;
      
      squares[col][row] = square;
    }
  }
}

void render_squares() {
  rectMode(CORNERS);
  textAlign(LEFT);
  
  for (int col = 0; col < squares.length; col++) {
    for (int row = 0; row < squares[0].length; row++) {
      
      if (squares[col][row].visible) {
        // Displaying square
        squares[col][row].display();
        
        // Displaying questions
        if (time_step >= 0) {
          if (squares[col][row].text.length() > 0) {
            fill(0);
            textSize(text_size);
            textLeading(line_spacing);
            text(squares[col][row].text, squares[col][row].tl.x + text_margin, squares[col][row].tl.y + text_margin, squares[col][row].br.x - text_margin, squares[col][row].br.y - text_margin);
          }
        }
      }
    }
  }
}

void update_cluster(int idx) {
  if (idx >= 0) {
    int square_row = questions_dst[idx] % squares[0].length;
    int square_col = int(questions_dst[idx] / squares[0].length);
    squares[square_col][square_row].text = squares[square_col][square_row].text + questions_list[idx] + "\n";
  }
}
