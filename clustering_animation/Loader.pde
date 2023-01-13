void load_trace() {
  trace = loadJSONObject(source_file);
  
  JSONArray temp_clusters = trace.getJSONArray("final_clustering");
  
  // Loading questions and dst
  JSONArray temp_list = trace.getJSONArray("questions_added");
  
  questions_list = new String[temp_list.size()];
  questions_dst = new int[temp_list.size()];
  
  for (int i = 0; i < temp_list.size(); i++) {
    questions_list[i] = temp_list.getString(i);
    questions_dst[i] = -1;
    
    for (int cluster_idx = 0; cluster_idx < temp_clusters.size(); cluster_idx++) {
      
      JSONArray temp_cluster = temp_clusters.getJSONArray(cluster_idx);
      
      for (int question_idx = 0; question_idx < temp_cluster.size(); question_idx++) {

        if (temp_cluster.getString(question_idx).equals(questions_list[i])) {
          questions_dst[i] = cluster_idx;
          break;
        }
      }
      if (questions_dst[i] >= 0) {
        break;
      }
    }
  }
}

String get_added_question(int idx) {
  return questions_list[idx];
}

int get_dst_cluster(int idx) {
  return questions_dst[idx];
}
