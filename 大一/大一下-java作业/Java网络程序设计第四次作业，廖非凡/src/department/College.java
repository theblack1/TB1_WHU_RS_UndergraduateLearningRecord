package department;

import java.util.Vector;

import Course.Course;
import main.Stage;
import person.Ruler;
import person.Teacher;

public class College extends Stage {
	public String type = "";
	public final static String T1 = "系";
	public final static String T2 = "实验中心";
	public final static String T3 = "研究所";

	public Vector<Course> coList = new Vector<Course>();
	public Vector<Teacher> trList = new Vector<Teacher>();

	// 一系列不同情况下使用的构造函数
	public College() {
	}

	public College(String type, String name, Ruler boss, Vector<Course> coList, Teacher firstTr) {
		if (type == T1) {// 确定类型！
			boss.level = Ruler.L31;
			this.type = T1;
		} else if (type == T2) {
			boss.level = Ruler.L32;
			this.type = T2;
		} else if (type == T3) {
			boss.level = Ruler.L33;
			this.type = T3;
		} else {
			System.out.println("Error:Type no found!");
			return;
		}

		this.name = name;
		this.boss = boss;

		for (Course s : coList) {
			AddCourse(s);
		}

		AddTeacher(firstTr);

		ShowMe();
	}

	public College(String type, String name, Ruler boss, Course firstCo, Teacher firstTr) {
		if (type == T1) {// 确定类型！
			boss.level = Ruler.L31;
			this.type = T1;
		} else if (type == T2) {
			boss.level = Ruler.L32;
			this.type = T2;
		} else if (type == T3) {
			boss.level = Ruler.L33;
			this.type = T3;
		} else {
			System.out.println("Error:Type no found!");
			return;
		}

		this.name = name;
		this.boss = boss;
		AddCourse(firstCo);
		AddTeacher(firstTr);

		ShowMe();
	}

	public College(String type, String name, Ruler boss, Vector<Course> coList) {
		if (type == T1) {// 确定类型！
			boss.level = Ruler.L31;
			this.type = T1;
		} else if (type == T2) {
			boss.level = Ruler.L32;
			this.type = T2;
		} else if (type == T3) {
			boss.level = Ruler.L33;
			this.type = T3;
		} else {
			System.out.println("Error:Type no found!");
			return;
		}

		this.name = name;
		this.boss = boss;
		for (Course s : coList) {
			AddCourse(s);
		}
	}

	public College(String type, String name, Ruler boss) {
		if (type == T1) {// 确定类型！
			boss.level = Ruler.L31;
			this.type = T1;
		} else if (type == T2) {
			boss.level = Ruler.L32;
			this.type = T2;
		} else if (type == T3) {
			boss.level = Ruler.L33;
			this.type = T3;
		} else {
			System.out.println("Error:Type no found!");
			return;
		}

		this.name = name;
		this.boss = boss;

		ShowMe();
	}

	// 添加课程和添加老师的函数
	public boolean AddCourse(Course course) {
		for (Course s : coList) {
			if (s == course)
				return false;
		}
		coList.add(course);
		return true;
	}

	public boolean AddTeacher(Teacher teacher) {
		for (Teacher s : trList) {
			if (s == teacher)
				return false;
		}
		trList.add(teacher);
		return true;
	}

	@Override

	public void ShowMe() {
		System.out.println();
		System.out.println(type + ":" + name + "\nDirector:" + boss.name);
		System.out.println("Course:");

		for (Course s : coList) {
			System.out.println("“" + s.name + "”");
		}
		System.out.println("Teacher:");
		for (Teacher s : trList) {
			System.out.println(s.name);
		}

	}

}
