#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace std;

// 互斥量，用于保护输出文件
std::mutex file_mutex;

// filter_csv_multithread(in_csv, output_folder, num_threads, start_time,
// end_time, dev_id);

// 线程函数:
void filter_csv(const string &in_csv, string out_folder, long long start_time,
                long long end_time, const string dev_id) {
  ifstream infile(in_csv);
  if (!infile.is_open()) {
    cerr << "无法打开原CSV文件: " << in_csv << endl;
    perror("错误信息");  // 打印出具体的错误信息
    return;
  }

  string line;
  long long line_cnt = 0;
  long long save_cnt = 0;
  while (getline(infile, line)) { // 对于每行:
    line_cnt++;
    istringstream ss(line);
    string id;    // current device id
    long long ts; // timestamp

    // 以逗号为间隔解析该行, 临时用dev_id存储(作为tmp_string)
    vector<string> tokens;
    while (getline(ss, id, ',')) {
      tokens.push_back(id);
    }

    // 从解析出的tokens获取变量
    id = tokens[0];
    ts = stoll(tokens.back());

    // 过滤条件
    if (id == dev_id && start_time <= ts && ts <= end_time) {
      save_cnt++;
      //写入对应子csv, 可能有冲突
      lock_guard<mutex> lock(file_mutex);

      string out_file = out_folder + "device_" + dev_id + ".csv"; // 子csv文件名
      ofstream outfile(out_file, ios::app); // 追加模式
      if (outfile.is_open()) {
        outfile << line << "\n";
        outfile.close();
        // 写入完成以后在原文件的行末尾追加一个标识符,下次不追加""
      } else {
        cerr << "无法打开输出文件: " << out_file << endl;
      }
      printf("处理第%lld行, 是该device第%lld个条目, 时间戳:%lld\n", line_cnt,
             save_cnt, ts);
    }
  }
}

// 按照时间(每24个小时)为线程分配任务。原文件时间格式是UNIX
void filter_csv_multithread(const string &in_csv, string out_folder,
                            long long start_time, long long end_time,
                            const string dev_id) {

  vector<thread> threads;
  long long duration = 24 * 60 * 60;

  for (long long cur_t = start_time; cur_t <= end_time;) {
    long long thread_t1 = cur_t;
    long long thread_t2 = min(cur_t + duration, end_time);
    // 创建线程并执行filter_csv函数
    threads.emplace_back(filter_csv, in_csv, out_folder, thread_t1, thread_t2,
                         dev_id);
  }

  // 等待所有线程完成
  for (auto &t : threads) {
    t.join();
  }

  cout << "Filter CSV file done!" << endl;
  return;
}

int main(int argc, char *argv[]) {
  if (argc != 4) {
    std::cerr << "用法: " << argv[0]
              << " <起始时间(UNIX格式)> <终止时间> <设备号>" << std::endl;
    return 1;
  }

  long long start_time = stoll(argv[1]);
  long long end_time = stoll(argv[2]);
  string dev_id = argv[3];

  string in_csv = "/home/data-7T/lzq/alibaba_block_traces_2020/io_traces.csv";
  int num_threads = 32; // 线程数量
  string output_folder =
      "/home/data-7T/lzq/alibaba_block_traces_2020/csvfix/"; // 输出文件夹

  filter_csv_multithread(in_csv, output_folder, start_time, end_time, dev_id);
}