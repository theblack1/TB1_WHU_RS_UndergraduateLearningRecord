package Course;

import java.util.Vector;

import main.CoRecord;
import main.Stage;
import person.Student;

public class Course extends Stage {
	int scores = -10;// 总分
	public int _courseNum = 0;// 课时数
	public Student monitor = new Student();// 班长
	public String teacher = "";// 任课老师

	public Vector<Student> stuList = new Vector<Student>();// 学生们

	public Course() {
	}

	// 添加课程信息
	public Course(String name, int score) {

		this.name = name;
		this.scores = score;
	}

	public Course(String name, int score, String teacher) {
		this.name = name;
		this.scores = score;
		this.teacher = teacher;
	}

	public Course(String name, int score, String teacher, Student monitor, Student firststu) {
		for (Student s : stuList)// 将学生加入课程
			AddStu(s);

		monitor.GetRecord(this).studyPower += 2;// 班长给分更高

		this.monitor = monitor;

		AddStu(firststu);
		this.name = name;
		this.scores = score;

		this.teacher = teacher;

		ShowMe();
	}

	public Course(String name, int score, String teacher, Student monitor, Vector<Student> stuList) {

		for (Student s : stuList)// 将学生加入课程
			AddStu(s);

		monitor.GetRecord(this).studyPower += 2;// 班长给分更高

		this.monitor = monitor;
		for (Student s : stuList) {
			AddStu(s);
		}

		this.name = name;
		this.scores = score;

		this.teacher = teacher;

		ShowMe();
	}

	// 添加学生
	public boolean AddStu(Student stu) {
		for (Student s : stuList) {// 检查是不是已经报名
			if (s == stu) {
				return false;
			}

		}
		stu.courseList.addElement(new CoRecord(stu.name, this));
		stuList.addElement(stu);
		return true;
	};

	// 判断是否是班长
	public boolean IsMonitor(final Student stu) {
		if (stu == monitor)
			return true;
		return false;
	}

	@Override
	public void ShowMe() {
		System.out.println("");
		System.out.println(
				"Course:" + name + "\nTotal score:" + scores + "\nTeacher:" + teacher + "\nMonior:" + monitor.name);
		System.out.println("Student:");
		for (Student s : stuList) {
			System.out.print(s.name);
			if (s == monitor)
				System.out.print("(Monitor)");
			System.out.println("");
		}

	}

}
