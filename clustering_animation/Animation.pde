class Animation {
  String question;
  int dst_cluster;
  float block_w;
  float block_h;
  float block_x;
  float block_y;
  int text_size = 42;
  int square_col;
  int square_row;
  int step = 0;
  float target_x;
  float target_y;
  
  float x_step;
  float y_step;
  float text_size_step;
  float block_w_step;
  float block_h_step;
  
  Animation() {
      this.question = get_added_question(time_step+1);
      this.dst_cluster = get_dst_cluster(time_step+1);
      this.block_w = (question.length() * text_size + text_size*1.5) * 0.4;
      this.block_h = this.text_size * 1.7;
      this.block_x = width/2;
      this.block_y = height/2;
      this.square_row = this.dst_cluster % squares[0].length;
      this.square_col = int(this.dst_cluster / squares[0].length);
      
      float[] center = squares[this.square_col][this.square_row].get_center();
      this.target_x = center[0];
      this.target_y = center[1];
      
      this.x_step = (this.block_x - this.target_x) / animation_steps;
      this.y_step = (this.block_y - this.target_y) / animation_steps;
      this.text_size_step = this.text_size / animation_steps;
      this.block_w_step = this.block_w / animation_steps;
      this.block_h_step = this.block_h / animation_steps;
  }
  
  void display() {
    rectMode(CENTER);
    fill(240, 220);
    int t_text_size = int(this.text_size - this.text_size_step * this.step);
    if (t_text_size < 0) {return;}
    float t_block_w = (question.length() * t_text_size + t_text_size*1.5) * 0.45;
    float t_block_h = t_text_size * 1.7;

    rect(this.block_x - this.x_step * this.step, this.block_y - this.y_step * this.step, t_block_w, t_block_h);
    
    textAlign(CENTER, CENTER);
    fill(0);
    textSize(int(t_text_size));
    text(question, this.block_x - this.x_step * this.step, this.block_y - this.y_step * this.step);
    
    if (this.step == animation_steps-1) {
      animating = false;
      background(bg_color);
      render_squares();
    }
    this.step++;
  }
}
