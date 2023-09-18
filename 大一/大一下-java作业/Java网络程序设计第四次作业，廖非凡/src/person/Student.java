package person;

import java.util.Random;
import java.util.Vector;

import Course.Course;
import main.CoRecord;

public class Student extends Person {

	public static int systemId = 1;// 学籍序号逐个上升,自动生成

	int id = -1;
	public String academy = "";
	public Vector<CoRecord> courseList = new Vector<CoRecord>();

	Random r = new Random();// 随机数生成器

	public Object study;

	public Student() {
	};

	// 填写基本信息
	public Student(String name, String academy) {
		this.name = name;
		this.academy = academy;
		id = Student.systemId++;
	};

	public Student(String name, String academy, Course firstcourse) {
		this.name = name;
		this.academy = academy;
		id = Student.systemId++;
		Addcourse(firstcourse);
		ShowMe();
	}

	public Student(String name, String academy, Vector<Course> coList) {
		this.name = name;
		this.academy = academy;
		id = Student.systemId++;
		for (Course s : coList) {
			Addcourse(s);
		}
		ShowMe();
	}

	// 添加课程
	public boolean Addcourse(Course course) {
		for (CoRecord s : courseList) {
			if (s.course == course) {
				return false;
			}
		}
		course.AddStu(this);
		return true;

	};

	// 听课
	public boolean attend(final Course course) {
		if (r.nextInt(100) < 2)// 每次每个人都有百分之一概率旷课，嘿嘿
			return false;
		GetRecord(course).studyPower++;
		return true;
	}

	// 做作业
	public int doHomework(final Course course) {
		CoRecord record = GetRecord(course);
		int ran1 = r.nextInt(100);
		if (ran1 < 11) {
			record.studyPower--;
			return 1;
		} // 不写作业
		else if (ran1 > 90)

		{
			record.studyPower++;
			return 2;
		} // 多学了一些
		else
			return 0;
	}

	// 取得对应成绩表
	public CoRecord GetRecord(final Course course) {
		for (CoRecord s : courseList) {
			if (s.course == course) {
				return s;
			}
		}
		// System.out.println("\n" + name + "同学没有选修" + course.name + "!");
		return new CoRecord();
	}

	@Override
	protected void Person(String name2) {
		super.name = name2;

	}

	@Override
	public void ShowMe() {
		System.out.println("");
		System.out.println("Student:");
		System.out.println("name:" + name + "\nAcademy:" + academy + "\nid:" + id);
		System.out.println("Courses:");
		for (CoRecord s : courseList) {
			System.out.println("“" + s.course.name + "”");
		}

	}

}
