package person;

import java.util.Vector;

import Course.Course;
import InterFaces.Study;
import main.CoRecord;

public class Undergraduate extends Student implements Study {

	public Undergraduate() {
	};

	public Undergraduate(String name, String academy) {
		this.name = name;
		this.academy = academy;
		id = Student.systemId++;
	};

	public Undergraduate(String name, String academy, Course firstcourse) {
		this.name = name;
		this.academy = academy;
		id = Student.systemId++;
		Addcourse(firstcourse);
		ShowMe();
	}

	public Undergraduate(String name, String academy, Vector<Course> coList) {
		this.name = name;
		this.academy = academy;
		id = Student.systemId++;
		for (Course s : coList) {
			Addcourse(s);
		}
		ShowMe();
	}

	public void ShowMe() {
		System.out.println("");
		System.out.println("本科生:");
		System.out.println("name:" + name + "\nAcademy:" + academy + "\nid:" + id);
		System.out.println("Courses:");
		for (CoRecord s : courseList) {
			System.out.println("“" + s.course.name + "”");
		}
	}

	@Override
	public void Study() {
		System.out.println("\n" + name + "理论学习");

	}

}
