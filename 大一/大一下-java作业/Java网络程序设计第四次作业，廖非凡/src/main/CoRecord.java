package main;

import Course.Course;

public class CoRecord {
	public String name = "";// 学生姓名
	public Course course = new Course();// 这门是什么课程

	public int skipTime = 0; // 记录旷课次数
	public int studyPower = 1; // 量化学习能力
	public int examScore = -1; // 期末考试成绩
	public int score = -1;// 总评成绩

	public CoRecord() {
	}

	public CoRecord(String name2, Course course2) {
		name = name2;
		course = course2;
	}

}
