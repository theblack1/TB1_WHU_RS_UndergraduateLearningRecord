package Course;

import java.util.Vector;

import person.Student;

public class PE extends Course {
	public final static String place = "体育馆";

	public PE() {
	};

	public PE(String name, int score) {

		this.name = name;
		this.scores = score;
	}

	public PE(String name, int score, String teacher) {
		this.name = name;
		this.scores = score;
		this.teacher = teacher;
	}

	public PE(String name, int score, String teacher, Student monitor, Student firststu) {
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

	public PE(String name, int score, String teacher, Student monitor, Vector<Student> stuList) {

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

	public void ShowMe() {
		System.out.println("");
		System.out.println("Place:" + place);
		System.out.println(
				"体育课:" + name + "\nTotal score:" + scores + "\nTeacher:" + teacher + "\nMonior:" + monitor.name);
		System.out.println("Student:");
		for (Student s : stuList) {
			System.out.print(s.name);
			if (s == monitor)
				System.out.print("(Monitor)");
			System.out.println("");
		}
	}
}
