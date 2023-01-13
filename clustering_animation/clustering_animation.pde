/**
 * This script creates the clustering animation from a JSON trace file 
 */
 
String source_file = "ensemble_10_gmm_diag.json"; //ensemble_10_gmm_diag

JSONObject trace;
String[] questions_list;
int[] questions_dst;

int time_step = -1;

// Squares Grid
Square[][] squares = new Square[2][5];  // square col, square row
final int border_spacing = 10;
final int intra_spacing = 10;
final int corner_radius = 4;
final color[] bg_squares = {
  #b3ffca,
  #ccb3ff,
  #99fdff,
  #ff8a80,
  #99ccff,
  #a9ff99,
  #ffdf80,
  #ff99cf,
  #d4ff4d,
  #c9eddc, 
};

// Graphics
int bg_color = 180;
int text_size = 15;
int line_spacing = text_size + 2;
int text_margin = 6;

// Animation
int delay_between_updates = 1000;
float animation_steps = 40;
int step_delay = 1000;
int initial_pause = 4000;
boolean animating = true;
Animation current_animation;
int animation_sequence = 0;

void setup() {
  //delay(6000);    // delay to start screen recorder
  rectMode(CORNERS);
  pixelDensity(1);
  size(1430, 750);
  frameRate(120);
  background(bg_color);
  
  load_trace();
  
  init_squares();
  render_squares();
  
  current_animation = new Animation();
}

void draw() {
  println(frameRate);
  background(bg_color);
  render_squares();
  
  if (animation_sequence == 0) {
    animation_sequence++;
    return;
    
  } else if (animation_sequence == 1) {
    delay(initial_pause*2);
    animation_sequence++;
    return;
  }
  
  if (animating && time_step < questions_list.length-1) {
    if (current_animation.step == 1) {
      delay(initial_pause);
    }
    current_animation.display();
    
  } else {
  
    if (time_step < questions_list.length-1) {
      time_step++;
      update_cluster(time_step);
    }
    
    if (time_step < questions_list.length-1) {
      animating = true;
      current_animation = new Animation();
    }
  }
}
